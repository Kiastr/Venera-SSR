package com.github.kiastr.venera_ssr.colorize

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

/**
 * 管理 ONNX Runtime 推理会话（session）生命周期。
 * 同一模型路径 + 同一执行后端（NNAPI/CPU）只加载一次并缓存复用
 * （纯 Dart 实现每图重建 session 是性能/稳定性瓶颈）。
 *
 * 移植自 AiColorize（com.kiastr.aicolorize.ModelManager）。
 */
class ModelManager(private val env: OrtEnvironment) {

    // 按 "模型路径|后端" 分别缓存 session。
    // 旧实现只持有单个 session，NNAPI↔CPU 交替使用（如上色走 NNAPI、放大回退 CPU）
    // 会反复 close + 重新加载模型 + 重编译计算图，翻页时表现为明显卡顿。
    private val sessions = HashMap<String, OrtSession>()

    private fun key(modelPath: String, useNnapi: Boolean) =
        "$modelPath|${if (useNnapi) "nnapi" else "cpu"}"

    @Synchronized
    fun getSession(modelPath: String, useNnapi: Boolean): OrtSession {
        val k = key(modelPath, useNnapi)
        sessions[k]?.let { return it }

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
        val s = env.createSession(modelPath, opts)
        sessions[k] = s
        return s
    }

    @Synchronized
    fun close() {
        for (s in sessions.values) {
            try {
                s.close()
            } catch (_: Exception) {
            }
        }
        sessions.clear()
    }
}
