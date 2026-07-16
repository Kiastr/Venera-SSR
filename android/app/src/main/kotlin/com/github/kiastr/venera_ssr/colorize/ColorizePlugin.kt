package com.github.kiastr.venera_ssr.colorize

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import org.opencv.android.OpenCVLoader
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors

/**
 * MethodChannel 桥接：Dart 侧通过 'colorize' 方法调用原生计算核心。
 * 单次调用完成：输入图片字节 -> 预处理 -> ONNX 推理 -> 后处理 -> 返回 PNG 字节。
 *
 * 移植自 AiColorize（com.kiastr.aicolorize.ColorizePlugin），改为接收/返回内存字节，
 * 避免中间落盘，更契合 Venera 的缓存式图片处理管线。
 */
class ColorizePlugin : MethodChannel.MethodCallHandler {

    companion object {
        const val CHANNEL = "com.github.kiastr.venera_ssr/colorize"

        fun registerWith(context: Context, messenger: BinaryMessenger) {
            val channel = MethodChannel(messenger, CHANNEL)
            channel.setMethodCallHandler(ColorizePlugin(context))
        }
    }

    private val context: Context
    private val engine = ColorizeEngine()
    private val executor = Executors.newSingleThreadExecutor()

    constructor(context: Context) {
        this.context = context
    }

    @Volatile
    private var openCvReady = false

    /** OpenCV 4.x Maven AAR 通过 OpenCVLoader.initDebug() 加载内置 .so（debug/release 均可用）。 */
    private fun ensureOpenCv() {
        if (!openCvReady) {
            synchronized(this) {
                if (!openCvReady) {
                    openCvReady = OpenCVLoader.initDebug()
                }
            }
        }
    }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        if (call.method == "resetSession") {
            // 丢弃已缓存的 ONNX 会话，下次推理按当前模型路径重新加载。
            // 切换/导入/删除模型后必须调用，否则 ModelManager 按路径缓存会沿用旧会话。
            engine.resetSession()
            result.success(null)
            return
        }
        if (call.method == "copyUri") {
            // 通过 ContentResolver 以有界分块（64KB）把 content URI / 文件路径拷贝到目标路径。
            // 用于导入外部模型文件：避免把 ~243MB 模型一次性读入内存（OOM），
            // 也避免 openRead 在 content URI 下不可靠的流式实现把文件拷坏。
            val uri = call.argument<String>("uri")
            val destPath = call.argument<String>("destPath")
            if (uri == null || destPath == null) {
                result.error("BAD_ARGS", "uri / destPath required", null)
                return
            }
            executor.execute {
                try {
                    val resolver = context.contentResolver
                    val srcUri = Uri.parse(uri)
                    val input = resolver.openInputStream(srcUri)
                    if (input == null) {
                        result.error("COPY_FAILED", "Cannot open input stream for $uri", null)
                        return@execute
                    }
                    var total = 0L
                    input.use {
                        val outFile = File(destPath)
                        outFile.parentFile?.mkdirs()
                        FileOutputStream(outFile).use { output ->
                            val buffer = ByteArray(64 * 1024)
                            var read: Int
                            while (input.read(buffer).also { read = it } != -1) {
                                output.write(buffer, 0, read)
                                total += read.toLong()
                            }
                            output.flush()
                        }
                    }
                    result.success(total)
                } catch (e: Throwable) {
                    // 捕获 Throwable（含 OOM/IO 异常），以 PlatformException 返回而非让进程崩溃
                    result.error("COPY_FAILED", e.message ?: "unknown error", null)
                }
            }
            return
        }
        if (call.method != "colorize") {
            result.notImplemented()
            return
        }
        val imageBytes = call.argument<ByteArray>("imageBytes")
        val modelPath = call.argument<String>("modelPath")
        val type = call.argument<String>("type") ?: "deoldify"
        val useNnapi = call.argument<Boolean>("useNnapi") ?: false
        val intensity = call.argument<Double>("intensity")?.toFloat() ?: 1.0f

        if (imageBytes == null || modelPath == null) {
            result.error("BAD_ARGS", "imageBytes / modelPath 不能为空", null)
            return
        }

        // 推理在后台线程执行，避免阻塞 UI（MethodChannel 结果回主线程派发）
        executor.execute {
            try {
                ensureOpenCv()
                if (!openCvReady) {
                    result.error("OPENCV_FAILED", "OpenCV native libs failed to load", null)
                    return@execute
                }
                val inputBitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                    ?: run {
                        result.error("DECODE_FAILED", "无法解码输入图片", null)
                        return@execute
                    }
                val outBitmap = engine.colorize(inputBitmap, modelPath, type, useNnapi, intensity)
                val baos = ByteArrayOutputStream()
                outBitmap.compress(Bitmap.CompressFormat.PNG, 100, baos)
                inputBitmap.recycle()
                outBitmap.recycle()
                result.success(baos.toByteArray())
            } catch (e: Throwable) {
                // 捕获 Throwable（含 Error/OOM），以 PlatformException 返回而非让进程崩溃
                result.error("COLORIZE_FAILED", e.message ?: "unknown error", null)
            }
        }
    }
}
