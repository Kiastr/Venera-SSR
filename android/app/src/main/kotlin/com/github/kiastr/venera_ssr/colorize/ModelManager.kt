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

    // 只为「当前模型路径」缓存多个后端（key = "nnapi" / "cpu"）。
    //
    // 旧实现只持有单个 session，同一模型在 NNAPI↔CPU 间切换（超分先试 NNAPI、
    // 失败回退 CPU）会反复 close + 重新加载 + 重编译计算图，表现为明显卡顿。
    // 因此这里允许同一模型的多后端并存。
    //
    // 但缓存必须限定在单一模型路径内：DeOldify 完整版约 243MB，
    // 若与超分模型的 session 同时长期驻留会显著抬高常驻内存。
    // 模型路径变化时释放旧路径的全部 session（与旧实现的释放时机一致）。
    private var currentPath: String? = null
    private val sessions = HashMap<String, OrtSession>()

    private fun backendKey(useNnapi: Boolean) = if (useNnapi) "nnapi" else "cpu"

    @Synchronized
    fun getSession(modelPath: String, useNnapi: Boolean): OrtSession {
        // 切换模型路径：释放旧模型的所有后端 session，避免多模型常驻内存
        if (currentPath != modelPath) {
            releaseAll()
            currentPath = modelPath
        }

        val k = backendKey(useNnapi)
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

    private fun releaseAll() {
        for (s in sessions.values) {
            try {
                s.close()
            } catch (_: Exception) {
            }
        }
        sessions.clear()
    }

    @Synchronized
    fun close() {
        releaseAll()
        currentPath = null
    }
}
