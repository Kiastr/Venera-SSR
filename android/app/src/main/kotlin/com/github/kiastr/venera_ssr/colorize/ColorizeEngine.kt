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
    // 对应 realesrgan 官方推理（img/255 -> RGB -> NCHW -> 推理 -> NCHW -> RGB -> *255）
    //  - 不归一化到 ImageNet 均值（animevideov3 仅 /255）
    //  - intensity 围绕中性灰 0.5 做对比缩放（默认 1.0 = 不变）
    //
    // 内存与后端策略：
    //  - 小图（整图输出 float buffer ≤ 120MB）走整图推理，沿用调用方后端
    //    （NNAPI 快且正确，与原行为一致）
    //  - 大图整图输出会超过 ART ~384MB 增长上限触发 OOM；改为分块推理。
    //    分块强制 CPU 后端：ONNX Runtime 的 NNAPI EP 对动态/小 tile 形状支持脆弱，
    //    会出现重编译卡顿且返回错误（黑屏）结果；CPU EP 对任意动态形状稳定正确。
    //  - 分块重叠 tilePad + 中心裁剪消除接缝；不持有整张 float32 输出（防 OOM）。
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
        val scale = 4
        val outH = h * scale
        val outW = w * scale
        // 整图输出 float buffer 字节数 ≈ 3 * 4H * 4W * 4
        val wholeOutBytes = 48L * h * w
        val SAFE = 120L * 1024 * 1024 // 120MB 余量，低于 ART 增长上限避免 OOM
        return if (wholeOutBytes <= SAFE) {
            // 小图：整图推理，沿用调用方后端（NNAPI 快且正确，与原行为一致）
            colorizeEsrganWhole(bgr, h, w, outH, outW, scale, session, intensity)
        } else {
            // 大图：强制 CPU 后端分块推理，规避 NNAPI 动态/小 tile 形状黑屏与重编译卡顿
            val cpuSession = modelManager.getSession(modelPath, false)
            colorizeEsrganTiled(bgr, h, w, outH, outW, scale, cpuSession, intensity)
        }
    }

    // 整图推理（小图，沿用调用方后端 NNAPI/CPU）
    private fun colorizeEsrganWhole(
        bgr: Mat, h: Int, w: Int, outH: Int, outW: Int, scale: Int,
        session: OrtSession, intensity: Float
    ): Bitmap {
        // BGR -> RGB（模型以 RGB 训练）
        val rgb = Mat()
        Imgproc.cvtColor(bgr, rgb, Imgproc.COLOR_BGR2RGB)
        // -> float32 0–1（/255，关键：animevideov3 仅归一化到 [0,1]）
        val rgbF = Mat()
        rgb.convertTo(rgbF, CvType.CV_32F, 1.0 / 255.0)

        // NCHW [1,3,H,W] 0–1
        val buf = ImageUtils.hwcToNchwFloatBuffer(rgbF)
        val inputName = session.inputNames.iterator().next()
        val tensor = OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, h.toLong(), w.toLong()))
        val results = session.run(Collections.singletonMap(inputName, tensor))
        // Result.get(int) 直接取 OnnxValue（避免 Optional 转型异常）
        val outBuf = (results.get(0) as OnnxTensor).floatBuffer // (1,3,4H,4W) NCHW RGB 0–1
        val outRgb = ImageUtils.nchwToHwcMat(outBuf, 3, outH, outW) // (4H,4W,3) RGB float 0–1
        tensor.close()
        results.close()

        // intensity：围绕 0.5 做对比缩放（默认 1.0 = 不变），在 float 空间计算避免截断误差
        if (intensity != 1.0f) {
            val half = Mat(outRgb.size(), outRgb.type(), Scalar(0.5))
            val centered = Mat()
            Core.subtract(outRgb, half, centered)
            Core.multiply(centered, Scalar(intensity.toDouble()), centered)
            Core.add(centered, half, outRgb)
            centered.release()
            half.release()
        }

        // *255 -> uint8（cvRound + 饱和截断 = clip[0,255]）
        val outRgbU8 = Mat()
        outRgb.convertTo(outRgbU8, CvType.CV_8U)
        outRgb.release()

        // RGB -> BGR（bgrMatToBitmap 内部再做 BGR2RGBA）
        val outBgr = Mat()
        Imgproc.cvtColor(outRgbU8, outBgr, Imgproc.COLOR_RGB2BGR)
        outRgbU8.release()

        val outBitmap = ImageUtils.bgrMatToBitmap(outBgr) // 内部 BGR2RGBA
        outBgr.release()
        bgr.release()
        rgb.release()
        rgbF.release()
        return outBitmap
    }

    // 分块推理（大图，强制 CPU 后端）
    private fun colorizeEsrganTiled(
        bgr: Mat, h: Int, w: Int, outH: Int, outW: Int, scale: Int,
        session: OrtSession, intensity: Float
    ): Bitmap {
        // BGR -> RGB -> float32 [0,1]
        val rgb = Mat()
        Imgproc.cvtColor(bgr, rgb, Imgproc.COLOR_BGR2RGB)
        val rgbF = Mat()
        rgb.convertTo(rgbF, CvType.CV_32F, 1.0 / 255.0)
        bgr.release()
        rgb.release()

        // 全图 uint8 输出（不持有整张 float32，防止 OOM）
        val outRgbU8 = Mat.zeros(outH, outW, CvType.CV_8UC(3))

        val tileSize = 384
        val tilePad = 16

        var yy = 0
        while (yy < h) {
            val yyEnd = minOf(h, yy + tileSize)
            val tileH = yyEnd - yy
            var xx = 0
            while (xx < w) {
                val xxEnd = minOf(w, xx + tileSize)
                val tileW = xxEnd - xx

                // 重叠 padding：超出图像边界时用反射填充，模型边界推理更稳
                val padTop = minOf(tilePad, yy)
                val padLeft = minOf(tilePad, xx)
                val padBottom = minOf(tilePad, h - yyEnd)
                val padRight = minOf(tilePad, w - xxEnd)

                val padded = Mat(rgbF, Rect(
                    xx - padLeft, yy - padTop,
                    tileW + padLeft + padRight,
                    tileH + padTop + padBottom
                ))
                val paddedTile = Mat()
                if (padTop > 0 || padBottom > 0 || padLeft > 0 || padRight > 0) {
                    Core.copyMakeBorder(padded, paddedTile, padTop, padBottom, padLeft, padRight, Core.BORDER_REFLECT)
                } else {
                    padded.copyTo(paddedTile)
                }

                // 推理：paddedTile (th,tw,3) float [0,1] -> outBuf (1,3,4th,4tw) NCHW
                val buf = ImageUtils.hwcToNchwFloatBuffer(paddedTile)
                val inputName = session.inputNames.iterator().next()
                val tensor = OnnxTensor.createTensor(env, buf,
                    longArrayOf(1, 3, paddedTile.height().toLong(), paddedTile.width().toLong()))
                val results = session.run(Collections.singletonMap(inputName, tensor))
                val outBuf = (results.get(0) as OnnxTensor).floatBuffer // (1,3,4th,4tw) NCHW RGB [0,1]
                val outTileH = paddedTile.height() * scale
                val outTileW = paddedTile.width() * scale
                val outTileRgb = ImageUtils.nchwToHwcMat(outBuf, 3, outTileH, outTileW) // (4th,4tw,3) float [0,1]
                tensor.close()
                results.close()

                // 中心裁剪：去掉 padding 对应的输出区域，消除接缝
                val cropTileRgb = Mat(outTileRgb,
                    Rect(padLeft * scale, padTop * scale, tileW * scale, tileH * scale))
                val cropTileRgbCopy = Mat()
                cropTileRgb.copyTo(cropTileRgbCopy)
                outTileRgb.release()

                // intensity：围绕 0.5 做对比缩放，在 float 空间计算避免截断误差
                if (intensity != 1.0f) {
                    val half = Mat(cropTileRgbCopy.size(), cropTileRgbCopy.type(), Scalar(0.5))
                    val centered = Mat()
                    Core.subtract(cropTileRgbCopy, half, centered)
                    Core.multiply(centered, Scalar(intensity.toDouble()), centered)
                    Core.add(centered, half, cropTileRgbCopy)
                    centered.release()
                    half.release()
                }

                // *255 -> uint8（cvRound + 饱和截断 = clip[0,255]）
                val cropTileU8 = Mat()
                cropTileRgbCopy.convertTo(cropTileU8, CvType.CV_8U)
                cropTileRgbCopy.release()

                // 写入整图输出 ROI
                val roi = Mat(outRgbU8, Rect(xx * scale, yy * scale, tileW * scale, tileH * scale))
                cropTileU8.copyTo(roi)
                cropTileU8.release()
                roi.release()
                paddedTile.release()

                xx = xxEnd
            }
            yy = yyEnd
        }

        rgbF.release()

        // RGB -> BGR（bgrMatToBitmap 内部再做 BGR2RGBA）
        val outBgr = Mat()
        Imgproc.cvtColor(outRgbU8, outBgr, Imgproc.COLOR_RGB2BGR)
        outRgbU8.release()

        val outBitmap = ImageUtils.bgrMatToBitmap(outBgr) // 内部 BGR2RGBA
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
