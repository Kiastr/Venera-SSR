package com.github.kiastr.venera_ssr.colorize

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer

/**
 * Bitmap ↔ OpenCV Mat ↔ ONNX FloatBuffer 的编解码工具。
 * 所有通道顺序、值域均对齐桌面版 cv2 行为。
 *
 * 移植自 AiColorize（com.kiastr.aicolorize.ImageUtils），经 Python 参考实现 + 真机模型端到端验证。
 */
object ImageUtils {

    /** Bitmap(ARGB_8888) -> OpenCV BGR Mat(uint8) */
    fun bitmapToBgrMat(bitmap: Bitmap): Mat {
        val rgba = Mat()
        Utils.bitmapToMat(bitmap, rgba) // 输出 RGBA 四通道
        val bgr = Mat()
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        rgba.release()
        return bgr
    }

    /** OpenCV BGR Mat -> Bitmap(ARGB_8888) */
    fun bgrMatToBitmap(bgr: Mat): Bitmap {
        val rgba = Mat()
        Imgproc.cvtColor(bgr, rgba, Imgproc.COLOR_BGR2RGBA)
        val bmp = Bitmap.createBitmap(bgr.width(), bgr.height(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgba, bmp)
        rgba.release()
        return bmp
    }

    /**
     * HWC Mat(CV_32F) -> NCHW FloatBuffer [C,H,W]
     * 与 numpy.transpose((2,0,1)) 再 expand_dims(0) 等价。
     */
    fun hwcToNchwFloatBuffer(mat: Mat): FloatBuffer {
        val h = mat.height()
        val w = mat.width()
        val c = mat.channels()
        val data = FloatArray(h * w * c)
        mat.get(0, 0, data) // OpenCV Mat 按 HWC row-major 存储
        val buf = FloatBuffer.allocate(c * h * w)
        for (ch in 0 until c) {
            for (y in 0 until h) {
                for (x in 0 until w) {
                    buf.put(data[(y * w + x) * c + ch])
                }
            }
        }
        buf.rewind()
        return buf
    }

    /**
     * NCHW FloatBuffer [C,H,W] -> HWC Mat(CV_32F)
     * 与 numpy.transpose(1,2,0) 等价。
     *
     * 注意：OpenCV Mat.put 期望 HWC 交错顺序，而模型输出是 NCHW（通道分离），
     * 必须显式转置，否则空间与通道会被打乱（表现为原图线条完好但散布随机彩点）。
     * 这是 Venera 原纯 Dart 实现之外，本项目在 AiColorize 上踩过并修复的根因。
     */
    fun nchwToHwcMat(buf: FloatBuffer, c: Int, h: Int, w: Int): Mat {
        buf.rewind()
        val nchw = FloatArray(c * h * w)
        buf.get(nchw) // NCHW: [ch][y][x] = nchw[ch*(h*w) + y*w + x]
        // 转置 NCHW -> HWC
        val hwc = FloatArray(c * h * w)
        for (ch in 0 until c) {
            for (y in 0 until h) {
                for (x in 0 until w) {
                    hwc[(y * w + x) * c + ch] = nchw[ch * (h * w) + y * w + x]
                }
            }
        }
        val mat = Mat(h, w, CvType.CV_32FC(c))
        mat.put(0, 0, hwc) // HWC 交错顺序
        return mat
    }
}
