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
import kotlin.math.roundToLong

/**
 * 上色/超分计算核心。三个函数严格对齐参考实现：
 *  - DeOldify 输入值域 0–255（不归一化），输出完整 BGR
 *  - DDColor  输入值域 0–1（/255），输出仅 ab 两通道，需与原图 L 拼接
 *  - ESRGAN   输入值域 0–1（/255），输出完整 RGB [1,3,scale·H,scale·W]；scale 由模型实际维度推导（见 getScale），当前权重 4x
 *  - OpenCV float-LAB 真实范围 L∈[0,100]、a,b∈[−128,127]，必须用 OpenCV 转换
 *
 * 移植自 AiColorize（com.kiastr.aicolorize.ColorizeEngine），经真机模型端到端验证。
 */
class ColorizeEngine {

    companion object {
        // NNAPI 输出与 CPU 参考逐通道 MAE 阈值（[0,1] 空间）；超过即判定偏色，整图回退 CPU。
        // NNAPI 在部分设备上仅做 fp16 近似，正常数值误差远小于此；真偏色（通道错位/染色）通常在 0.1+。
        private const val NNAPI_COLOR_TOL = 0.04
        // 整图通道均值偏色阈值（uint8 空间 [0,255]）。仅 NNAPI 路径、全图推理完成后做一次：
        // 真偏色表现为 R 通道独高、G≈B，R-G 均值远超正常图（正常动漫各色平均后 R≈G≈B，典型 < 20）；
        // 实测染红设备 R-G 均值常 > 60，留足余量取 30。首 tile 的 MAE 检查会被顶部黑边蒙混
        // （NNAPI 对全黑输入也输出近乎全黑，MAE≈0），故以全图均值作兜底。
        private const val NNAPI_COLOR_TOL_RGB = 30.0
    }

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
    // ESRGAN (Real-ESRGAN animevideov3)：输入 0–1，输出完整 RGB；放大倍数由模型实际维度推导（见 getScale），当前权重为 4x
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
            // 放大倍数从模型实际输入/输出维度推导（探测一次并缓存），不再硬编码 4——
            // 换 2x 模型时自动适配，无需改代码。用 CPU 会话探测，与 NNAPI 是否偏色无关。
            val cpuSession = modelManager.getSession(modelPath, false)
            val scale = getScale(modelPath, cpuSession)
            if (!useNnapi) {
                // CPU：输出恒正确，无需自检
                return colorizeEsrganFixedTile(rgbF, h, w, session, intensity, scale, cpuRef = null)!!
            }
            // NNAPI：首 tile 与 CPU 参考对比，拦截后端返回的错误颜色（典型如整体偏红）。
            // 旧逻辑只检测"全黑"，漏掉了"偏色但仍非黑"的损坏——某些设备的 NNAPI
            // 会把 Real-ESRGAN 输出染红而非归零，表现就是超分后发红。
            val out = colorizeEsrganFixedTile(rgbF, h, w, session, intensity, scale, cpuRef = cpuSession)
            if (out != null) return out

            // NNAPI 不可信 -> 整图切 CPU 重跑
            return colorizeEsrganFixedTile(rgbF, h, w, cpuSession, intensity, scale, cpuRef = null)!!
        } finally {
            rgbF.release()
        }
    }

    /**
     * 固定尺寸分块推理。
     * 返回 null 表示 NNAPI 首 tile 自检未过（全黑或偏色，后端不可信），由调用方回退 CPU。
     * cpuRef 非空时开启自检；为 null 时（CPU 路径或回退重跑）不做自检。
     */
    private fun colorizeEsrganFixedTile(
        rgbF: Mat, h: Int, w: Int,
        session: OrtSession, intensity: Float, scale: Int, cpuRef: OrtSession?
    ): Bitmap? {
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

                // 后端健全性自检（仅 NNAPI 首 tile 开启，cpuRef 即 CPU 参考会话）
                if (cpuRef != null && firstTile) {
                    val inMean = Core.mean(tile).`val`.let { (it[0] + it[1] + it[2]) / 3.0 }
                    val outMean = Core.mean(outTileRgb).`val`.let { (it[0] + it[1] + it[2]) / 3.0 }
                    // 1) 全黑：输入非黑但输出全黑 => 后端异常（原自检逻辑保留）
                    if (inMean > 0.05 && outMean < 0.01) {
                        outTileRgb.release()
                        tile.release()
                        outRgbU8.release()
                        return null
                    }
                    // 2) 偏色：与 CPU 参考 tile 比对，逐通道 MAE 超阈值即判定后端返回了错误颜色
                    //    （典型如整体偏红）。NNAPI 在部分设备上仅做 fp16 近似，正常数值误差远小于阈值；
                    //    真偏色（通道错位/染色）的 MAE 通常在 0.1+，阈值取 0.04（≈10/255）留足余量。
                    try {
                        val refBuf = ImageUtils.hwcToNchwFloatBuffer(tile)
                        val refTensor = OnnxTensor.createTensor(
                            env, refBuf, longArrayOf(1, 3, tileIn.toLong(), tileIn.toLong())
                        )
                        val refRes = cpuRef.run(Collections.singletonMap(inputName, refTensor))
                        val refOut = (refRes.get(0) as OnnxTensor).floatBuffer
                        val refTileRgb = ImageUtils.nchwToHwcMat(refOut, 3, tileIn * scale, tileIn * scale)
                        refTensor.close()
                        refRes.close()
                        val mae = maxChannelMae(outTileRgb, refTileRgb)
                        refTileRgb.release()
                        if (mae > NNAPI_COLOR_TOL) {
                            outTileRgb.release()
                            tile.release()
                            outRgbU8.release()
                            return null
                        }
                    } catch (_: Exception) {
                        // 参考比对异常则不阻断 NNAPI（退化为仅"全黑"自检）
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

        // 整图偏色自检（仅 NNAPI 路径，cpuRef != null）：首 tile 的逐通道 MAE 检查会被顶部
        // 黑边蒙混（NNAPI 对全黑输入也输出近乎全黑，MAE≈0，漏检偏色），故改在全图推理完成
        // 后做通道均值检查。真偏色表现为 R 通道独高、G≈B，R-G 均值远高于正常图；正常动漫
        // 各色平均后 R≈G≈B，远小于阈值。命中即整图作废，由调用方切 CPU 重跑。
        if (cpuRef != null) {
            val m = Core.mean(outRgbU8).`val`
            if (m[0] - m[1] > NNAPI_COLOR_TOL_RGB || m[0] - m[2] > NNAPI_COLOR_TOL_RGB) {
                outRgbU8.release()
                return null
            }
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

    /**
     * 两个 float HWC Mat 的逐通道 mean absolute error，取三通道中的最大值。
     * 用于 NNAPI 输出与 CPU 参考的偏色比对：空间细节的 fp16 噪声会相互抵消，
     * 但通道错位/染色的 MAE 会显著偏高。
     */
    private fun maxChannelMae(a: Mat, b: Mat): Double {
        val d = Mat()
        Core.absdiff(a, b, d)
        val m = Core.mean(d).`val` // 三通道各自的 mean abs error
        d.release()
        return maxOf(m[0], m[1], m[2])
    }

    /**
     * 从模型实际前向的输入/输出维度推导放大倍数（scale）。
     * 用一次极小的 dummy 前向（CPU 会话，与 NNAPI 是否偏色无关）测量输出/输入的空间比值，
     * 按 modelPath 缓存。这样换 2x/4x 权重时无需改代码即可自动适配，当前 animevideov3 为 4x。
     *
     * 用输出 NCHW float buffer 的长度反推边长，避免依赖各 onnxruntime 版本输出 shape API 的差异。
     */
    private val scaleCache = HashMap<String, Int>()

    private fun getScale(modelPath: String, cpuSession: OrtSession): Int {
        synchronized(scaleCache) {
            scaleCache[modelPath]?.let { return it }
        }
        val probeIn = 64 // 正方形小图，输出必为 (probeIn*scale)²，避免动态维度干扰
        val inputName = cpuSession.inputNames.iterator().next()
        // 全零 dummy 即可：放大倍数只取决于形状，与具体数值无关
        val dummy = Mat.zeros(probeIn, probeIn, CvType.CV_32FC(3))
        val buf = ImageUtils.hwcToNchwFloatBuffer(dummy)
        dummy.release()
        val tensor = OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, probeIn.toLong(), probeIn.toLong()))
        val results = cpuSession.run(Collections.singletonMap(inputName, tensor))
        val outBuf = (results.get(0) as OnnxTensor).floatBuffer
        val total = outBuf.remaining().toLong()       // 1*3*outH*outW
        val px = total / 3                            // outH*outW
        val side = kotlin.math.sqrt(px.toDouble()).roundToLong() // 输入正方形 => 输出正方形
        tensor.close()
        results.close()
        val scale = if (side > 0 && side % probeIn == 0L) (side / probeIn).toInt() else 4
        synchronized(scaleCache) {
            scaleCache[modelPath] = scale
        }
        return scale
    }

    /// 丢弃已缓存的 ONNX 会话（仅关闭 session，不释放 env）。
    /// 模型切换/导入/删除后调用，使下次推理按当前路径重新加载。
    fun resetSession() {
        modelManager.close()
    }
}
