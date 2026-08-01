package com.github.kiastr.venera_ssr.colorize

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.util.Collections

/**
 * 上色/超分计算核心。三个函数严格对齐参考实现：
 *  - DeOldify 输入值域 0–255（不归一化），输出完整 BGR
 *  - DDColor  输入值域 0–1（/255），输出仅 ab 两通道，需与原图 L 拼接
 *  - ESRGAN   输入值域 0–1（/255），输出完整 RGB [1,3,4H,4W]，固定 4x
 *  - OpenCV float-LAB 真实范围 L∈[0,100]、a,b∈[−128,127]，必须用 OpenCV 转换
 *
 * 移植自 AiColorize（com.kiastr.aicolorize.ColorizeEngine），经真机模型端到端验证。
 */
class ColorizeEngine {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val modelManager = ModelManager(env)

    /**
     * 单次处理：输入已解码的 Bitmap，返回处理后的 Bitmap。
     * 不负责输入 Bitmap 的回收（由调用方持有），仅回收内部 Mat 与输出 Bitmap 之外的中间对象。
     */
    fun colorize(
        inputBitmap: Bitmap,
        modelPath: String,
        type: String,
        useNnapi: Boolean,
        intensity: Float
    ): Bitmap {
        // Utils.bitmapToMat 要求 ARGB_8888；非该配置会产出异常/空 Mat（曾导致 resize 空源崩溃）
        val (working, needsRecycle) = if (inputBitmap.config == Bitmap.Config.ARGB_8888) {
            Pair(inputBitmap, false)
        } else {
            Pair(inputBitmap.copy(Bitmap.Config.ARGB_8888, false), true)
        }
        if (working.width <= 0 || working.height <= 0) {
            if (needsRecycle) working.recycle()
            throw IllegalArgumentException("图片尺寸无效: ${working.width}x${working.height}")
        }
        val session = modelManager.getSession(modelPath, useNnapi)
        val out = when (type) {
            "ddcolor" -> colorizeDdcolor(working, session, intensity)
            "esrgan" -> colorizeEsrgan(working, session, modelPath, useNnapi, intensity)
            else -> colorizeDeoldify(working, session, intensity)
        }
        if (needsRecycle) working.recycle()
        return out
    }

    // ----------------------------------------------------------------
    // DeOldify：输入 0–255，输出完整 BGR，再与原图 L 通道在 LAB 合并
    // 对应 colorize_deoldify（py_ref_impl.colorize_deoldify）
    // ----------------------------------------------------------------
    private fun colorizeDeoldify(inputBitmap: Bitmap, session: OrtSession, intensity: Float): Bitmap {
        val originalBgr = ImageUtils.bitmapToBgrMat(inputBitmap) // (H,W,3) BGR uint8
        val h = originalBgr.height()
        val w = originalBgr.width()

        // target_l = 原图 BGR 的 B 通道（cv2.split 第一个）
        val targetL = Mat()
        Core.extractChannel(originalBgr, targetL, 0)

        // gray = BGR2GRAY
        val gray = Mat()
        Imgproc.cvtColor(originalBgr, gray, Imgproc.COLOR_BGR2GRAY)
        // gray_rgb = GRAY2RGB
        val grayRgb = Mat()
        Imgproc.cvtColor(gray, grayRgb, Imgproc.COLOR_GRAY2RGB)
        // resize(256,256)
        val input256 = Mat()
        Imgproc.resize(grayRgb, input256, Size(256.0, 256.0))
        // -> float32 0–255（不 /255！）
        val inputF = Mat()
        input256.convertTo(inputF, CvType.CV_32F)

        // NCHW [1,3,256,256] 0–255
        val buf = ImageUtils.hwcToNchwFloatBuffer(inputF)
        val inputName = session.inputNames.iterator().next()
        val tensor = OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, 256, 256))
        val results = session.run(Collections.singletonMap(inputName, tensor))
        // Result.get(String) 返回 Optional<OnnxValue>（直接转型 OnnxTensor 会 ClassCastException），
        // 改用 get(int) 直接取 OnnxValue
        val outBuf = (results.get(0) as OnnxTensor).floatBuffer // (1,3,256,256) NCHW BGR 0–255
        val colorized256 = ImageUtils.nchwToHwcMat(outBuf, 3, 256, 256) // (256,256,3) BGR
        tensor.close()
        results.close()

        // BGR2RGB -> uint8
        val colorizedRgb = Mat()
        Imgproc.cvtColor(colorized256, colorizedRgb, Imgproc.COLOR_BGR2RGB)
        colorized256.release()
        val colorizedUint8 = Mat()
        colorizedRgb.convertTo(colorizedUint8, CvType.CV_8U) // 饱和截断
        colorizedRgb.release()

        // resize 回原尺寸
        val colorizedFull = Mat()
        Imgproc.resize(colorizedUint8, colorizedFull, Size(w.toDouble(), h.toDouble()))
        colorizedUint8.release()

        // GaussianBlur(13,13)
        val blurred = Mat()
        Imgproc.GaussianBlur(colorizedFull, blurred, Size(13.0, 13.0), 0.0)
        colorizedFull.release()

        // ★ 刻意对齐原版：colorized 此时是 RGB，但原代码用 COLOR_BGR2LAB 处理（即把 RGB 当 BGR）
        val lab = Mat()
        Imgproc.cvtColor(blurred, lab, Imgproc.COLOR_BGR2Lab)
        blurred.release()

        val channels = ArrayList<Mat>()
        Core.split(lab, channels)
        var a = channels[1]
        var b = channels[2]

        // intensity：在 A/B 通道上围绕中性灰 128 做缩放（默认 1.0 = 不变），保留 L 不变。
        // 在 float 空间计算后截断回 uint8，避免 uint8 直接运算的饱和误差。
        if (intensity != 1.0f) {
            val aF = Mat()
            val bF = Mat()
            a.convertTo(aF, CvType.CV_32F)
            b.convertTo(bF, CvType.CV_32F)
            Core.subtract(aF, Scalar(128.0), aF)
            Core.multiply(aF, Scalar(intensity.toDouble()), aF)
            Core.add(aF, Scalar(128.0), aF)
            aF.convertTo(a, CvType.CV_8U) // 写回 a（即 channels[1]）
            Core.subtract(bF, Scalar(128.0), bF)
            Core.multiply(bF, Scalar(intensity.toDouble()), bF)
            Core.add(bF, Scalar(128.0), bF)
            bF.convertTo(b, CvType.CV_8U) // 写回 b（即 channels[2]）
            aF.release()
            bF.release()
        }

        val merged = ArrayList<Mat>()
        merged.add(targetL)
        merged.add(a)
        merged.add(b)
        val resultLab = Mat()
        Core.merge(merged, resultLab)
        lab.release()
        channels[0].release()
        a.release()
        b.release()

        // LAB2BGR
        val resultBgr = Mat()
        Imgproc.cvtColor(resultLab, resultBgr, Imgproc.COLOR_Lab2BGR)
        resultLab.release()
        targetL.release()

        val outBitmap = ImageUtils.bgrMatToBitmap(resultBgr) // 内部 BGR2RGBA
        resultBgr.release()
        originalBgr.release()
        gray.release()
        grayRgb.release()
        input256.release()
        inputF.release()
        return outBitmap
    }

    // ----------------------------------------------------------------
    // DDColor：输入 0–1，输出仅 ab 两通道，需与原图 L 拼接
    // 对应 colorize_ddcolor_tiny（py_ref_impl.colorize_ddcolor）
    // ----------------------------------------------------------------
    private fun colorizeDdcolor(inputBitmap: Bitmap, session: OrtSession, intensity: Float): Bitmap {
        val bgr = ImageUtils.bitmapToBgrMat(inputBitmap) // BGR uint8
        val h = bgr.height()
        val w = bgr.width()

        // img_norm = bgr / 255
        val imgNorm = Mat()
        bgr.convertTo(imgNorm, CvType.CV_32F, 1.0 / 255.0)

        // orig_l = BGR2Lab[:,:,:1]（float LAB, L∈[0,100]）
        val labFull = Mat()
        Imgproc.cvtColor(imgNorm, labFull, Imgproc.COLOR_BGR2Lab)
        imgNorm.release()
        val origL = Mat()
        Core.extractChannel(labFull, origL, 0)
        labFull.release()

        // img_resized 256
        val imgResized = Mat()
        Imgproc.resize(imgNorm, imgResized, Size(256.0, 256.0))
        // img_l from resized
        val labResized = Mat()
        Imgproc.cvtColor(imgResized, labResized, Imgproc.COLOR_BGR2Lab)
        val imgL = Mat()
        Core.extractChannel(labResized, imgL, 0)
        labResized.release()
        imgResized.release()

        // gray_lab = concat(img_l, 0, 0)
        val zeros = Mat(imgL.size(), imgL.type(), Scalar(0.0))
        val grayLabList = ArrayList<Mat>()
        grayLabList.add(imgL)
        grayLabList.add(zeros)
        grayLabList.add(zeros)
        val grayLab = Mat()
        Core.merge(grayLabList, grayLab)
        // gray_rgb = LAB2RGB（0–1）
        val grayRgb = Mat()
        Imgproc.cvtColor(grayLab, grayRgb, Imgproc.COLOR_Lab2RGB)
        zeros.release()

        // NCHW [1,3,256,256] 0–1
        val buf = ImageUtils.hwcToNchwFloatBuffer(grayRgb)
        val inputName = session.inputNames.iterator().next()
        val tensor = OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, 256, 256))
        val results = session.run(Collections.singletonMap(inputName, tensor))
        // Result.get(String) 返回 Optional<OnnxValue>；改用 get(int)
        val outBuf = (results.get(0) as OnnxTensor).floatBuffer // (1,2,256,256) NCHW ab
        val ab256 = ImageUtils.nchwToHwcMat(outBuf, 2, 256, 256) // (256,256,2) ab
        tensor.close()
        results.close()

        // resize ab 回原尺寸
        val abFull = Mat()
        Imgproc.resize(ab256, abFull, Size(w.toDouble(), h.toDouble()))
        ab256.release()

        // intensity：在 ab 通道上围绕 0 做缩放（默认 1.0 = 不变）
        if (intensity != 1.0f) {
            Core.multiply(abFull, Scalar(intensity.toDouble()), abFull)
        }

        // output_lab = concat(orig_l, abFull)
        val outLabList = ArrayList<Mat>()
        outLabList.add(origL)
        outLabList.add(abFull)
        val outLab = Mat()
        Core.merge(outLabList, outLab)
        origL.release()
        abFull.release()

        // output_bgr = LAB2BGR（float 0–1）
        val outBgr = Mat()
        Imgproc.cvtColor(outLab, outBgr, Imgproc.COLOR_Lab2BGR)
        outLab.release()

        // (outBgr * 255).round().clip(0,255).astype(uint8)
        // OpenCV convertTo(CV_8U) 内部用 cvRound 四舍五入 + 饱和截断（已实证），
        // 因此只需 *255 后直接 convertTo，切勿再 +0.5（会变成双重舍入、整体偏亮 ~1）
        val scaled = Mat()
        Core.multiply(outBgr, Scalar(255.0), scaled)
        val outUint8 = Mat()
        scaled.convertTo(outUint8, CvType.CV_8U) // cvRound + 饱和截断 = clip[0,255]
        scaled.release()
        outBgr.release()

        val outBitmap = ImageUtils.bgrMatToBitmap(outUint8) // 内部 BGR2RGBA
        outUint8.release()
        bgr.release()
        grayLab.release()
        grayRgb.release()
        imgL.release()
        return outBitmap
    }

    // ----------------------------------------------------------------
    // ESRGAN (Real-ESRGAN animevideov3)：输入 0–1，固定 4x 输出完整 RGB
    // 对应 realesrgan 官方推理（img/255 -> RGB -> NCHW -> 推理 -> NCHW -> *255）
    //  - 不归一化到 ImageNet 均值（animevideov3 仅 /255）
    //  - intensity 围绕中性灰 0.5 做对比缩放（默认 1.0 = 不变）
    //
    // 固定尺寸分块策略（NNAPI 友好）：
    //  - 所有 tile（含边界残块）一律 padding 到恒定 TILE_IN×TILE_IN 再送入模型。
    //    ONNX Runtime 的 NNAPI EP 在模型含动态维度时按首次实际形状编译计算图；
    //    形状每变一次就重编译一次，是此前"大图极慢"的主因。输入形状恒定后
    //    整个会话只编译一次，后续 tile 直接复用，可全程走 GPU/NPU。
    //  - ART 堆峰值恒定 = 3*(4*TILE_IN)^2*4 字节 ≈ 28MB（与原图尺寸无关），
    //    彻底消除此前按图尺寸增长导致的 OOM。
    //  - 核心区在 tile 内恒定从 (PAD,PAD) 起，输出按 PAD*scale 中心裁剪，消除接缝。
    //  - NNAPI 输出健全性自检：首个 tile 若"输入非黑、输出全黑"，判定后端异常，
    //    自动整图切 CPU 重跑（CPU EP 对任意形状稳定正确）。
    // ----------------------------------------------------------------
    private fun colorizeEsrgan(
        inputBitmap: Bitmap,
        session: OrtSession,
        modelPath: String,
        useNnapi: Boolean,
        intensity: Float
    ): Bitmap {
        val bgr = ImageUtils.bitmapToBgrMat(inputBitmap) // (H,W,3) BGR uint8
        val h = bgr.height()
        val w = bgr.width()

        // BGR -> RGB（模型以 RGB 训练）-> float32 [0,1]（animevideov3 仅 /255）
        val rgb = Mat()
        Imgproc.cvtColor(bgr, rgb, Imgproc.COLOR_BGR2RGB)
        val rgbF = Mat()
        rgb.convertTo(rgbF, CvType.CV_32F, 1.0 / 255.0)
        bgr.release()
        rgb.release()

        try {
            // 首选调用方指定的后端；NNAPI 时开启输出自检
            val out = colorizeEsrganFixedTile(rgbF, h, w, session, intensity, checkSanity = useNnapi)
            if (out != null) return out

            // NNAPI 输出异常 -> 整图切 CPU 重跑
            val cpuSession = modelManager.getSession(modelPath, false)
            return colorizeEsrganFixedTile(rgbF, h, w, cpuSession, intensity, checkSanity = false)!!
        } finally {
            rgbF.release()
        }
    }

    /**
     * 固定尺寸分块推理。
     * 返回 null 表示 checkSanity 开启且首个 tile 输出异常（后端不可信），由调用方回退。
     */
    private fun colorizeEsrganFixedTile(
        rgbF: Mat, h: Int, w: Int,
        session: OrtSession, intensity: Float, checkSanity: Boolean
    ): Bitmap? {
        val scale = 4
        val outH = h * scale
        val outW = w * scale

        // 固定输入边长。384 兼顾单 tile 计算量与 ART 堆峰值（输出 1536² ≈ 28MB）
        val tileIn = 384
        val pad = 16                    // 每边重叠感受野，裁剪后消除接缝
        val core = tileIn - 2 * pad     // 每个 tile 实际贡献的有效边长 = 352

        // 全图 uint8 输出（native 内存，不占 ART 堆）
        val outRgbU8 = Mat.zeros(outH, outW, CvType.CV_8UC(3))
        val inputName = session.inputNames.iterator().next()
        var firstTile = true

        var y = 0
        while (y < h) {
            val coreH = minOf(core, h - y)
            var x = 0
            while (x < w) {
                val coreW = minOf(core, w - x)

                // 源区域（核心区 + 四周 pad），clamp 到图像内
                val sx0 = maxOf(0, x - pad)
                val sy0 = maxOf(0, y - pad)
                val sx1 = minOf(w, x + coreW + pad)
                val sy1 = minOf(h, y + coreH + pad)

                // 补齐到恒定 tileIn×tileIn：越界侧与右/下不足部分用边缘复制填充
                // （BORDER_REPLICATE 对任意 border 大小安全；BORDER_REFLECT 在
                //  border >= 源边长时行为受限，小图会出问题）
                val padLeft = maxOf(0, pad - x)
                val padTop = maxOf(0, pad - y)
                val padRight = tileIn - padLeft - (sx1 - sx0)
                val padBottom = tileIn - padTop - (sy1 - sy0)

                val src = Mat(rgbF, Rect(sx0, sy0, sx1 - sx0, sy1 - sy0))
                val tile = Mat()
                Core.copyMakeBorder(
                    src, tile, padTop, padBottom, padLeft, padRight, Core.BORDER_REPLICATE
                )
                src.release()

                // 推理：tile (tileIn,tileIn,3) float[0,1] -> (1,3,4*tileIn,4*tileIn) NCHW
                val buf = ImageUtils.hwcToNchwFloatBuffer(tile)
                val tensor = OnnxTensor.createTensor(
                    env, buf, longArrayOf(1, 3, tileIn.toLong(), tileIn.toLong())
                )
                val results = session.run(Collections.singletonMap(inputName, tensor))
                val outBuf = (results.get(0) as OnnxTensor).floatBuffer
                val outTileRgb = ImageUtils.nchwToHwcMat(outBuf, 3, tileIn * scale, tileIn * scale)
                tensor.close()
                results.close()

                // 首个 tile 自检：输入非纯黑但输出全黑 => 后端返回了无效结果
                if (checkSanity && firstTile) {
                    val inMean = Core.mean(tile).`val`.let { (it[0] + it[1] + it[2]) / 3.0 }
                    val outMean = Core.mean(outTileRgb).`val`.let { (it[0] + it[1] + it[2]) / 3.0 }
                    if (inMean > 0.05 && outMean < 0.01) {
                        outTileRgb.release()
                        tile.release()
                        outRgbU8.release()
                        return null
                    }
                }
                firstTile = false
                tile.release()

                // 核心区在 tile 内恒定从 (pad,pad) 起（已验证与 x/y 是否贴边无关）
                val cropRgb = Mat(
                    outTileRgb,
                    Rect(pad * scale, pad * scale, coreW * scale, coreH * scale)
                )
                val cropCopy = Mat()
                cropRgb.copyTo(cropCopy)
                outTileRgb.release()

                // intensity：围绕 0.5 做对比缩放，在 float 空间计算避免截断误差
                if (intensity != 1.0f) {
                    val half = Mat(cropCopy.size(), cropCopy.type(), Scalar(0.5))
                    val centered = Mat()
                    Core.subtract(cropCopy, half, centered)
                    Core.multiply(centered, Scalar(intensity.toDouble()), centered)
                    Core.add(centered, half, cropCopy)
                    centered.release()
                    half.release()
                }

                // *255 -> uint8（convertTo 的 alpha=255：cvRound + 饱和截断 = clip[0,255]）
                val cropU8 = Mat()
                cropCopy.convertTo(cropU8, CvType.CV_8U, 255.0)
                cropCopy.release()

                // 写入整图输出 ROI
                val roi = Mat(outRgbU8, Rect(x * scale, y * scale, coreW * scale, coreH * scale))
                cropU8.copyTo(roi)
                cropU8.release()
                roi.release()

                x += coreW
            }
            y += coreH
        }

        // RGB -> BGR（bgrMatToBitmap 内部再做 BGR2RGBA）
        val outBgr = Mat()
        Imgproc.cvtColor(outRgbU8, outBgr, Imgproc.COLOR_RGB2BGR)
        outRgbU8.release()

        val outBitmap = ImageUtils.bgrMatToBitmap(outBgr)
        outBgr.release()
        return outBitmap
    }

    fun close() {
        modelManager.close()
    }

    /// 丢弃已缓存的 ONNX 会话（仅关闭 session，不释放 env）。
    /// 模型切换/导入/删除后调用，使下次推理按当前路径重新加载。
    fun resetSession() {
        modelManager.close()
    }
}
