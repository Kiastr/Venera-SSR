package com.github.kiastr.venera_ssr.colorize

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

/**
 * 管理 ONNX Runtime 推理会话（session）生命周期。
 * 同一模型路径只加载一次并缓存复用（纯 Dart 实现每图重建 session 是性能/稳定性瓶颈）。
 *
 * 移植自 AiColorize（com.kiastr.aicolorize.ModelManager）。
 */
class ModelManager(private val env: OrtEnvironment) {

    @Volatile
    private var session: OrtSession? = null

    @Volatile
    private var currentPath: String? = null

    @Synchronized
    fun getSession(modelPath: String, useNnapi: Boolean): OrtSession {
        if (session == null || currentPath != modelPath) {
            session?.close()
            val opts = OrtSession.SessionOptions()
            if (useNnapi) {
                try {
                    // NNAPI 统一抽象 CPU/GPU/NPU；不支持时回退 CPU
                    opts.addNnapi()
                } catch (e: Exception) {
                    opts.addCPU(true)
                }
            } else {
                opts.addCPU(true)
            }
            session = env.createSession(modelPath, opts)
            currentPath = modelPath
        }
        return session!!
    }

    @Synchronized
    fun close() {
        session?.close()
        session = null
        currentPath = null
    }
}
