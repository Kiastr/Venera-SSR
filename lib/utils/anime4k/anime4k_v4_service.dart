import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import 'package:venera/foundation/app.dart';
import 'package:venera/foundation/log.dart';

import 'anime4k_v4_model_manager.dart';

/// Anime4K v4 超分服务（带模型版本）
///
/// 基于 Real-ESRGAN(animevideov3) ONNX 模型，经原生（Kotlin + ONNX Runtime + NNAPI GPU）
/// 完成超分辨率，固定 4x 放大。设计严格对齐 [ColorizationService]：
///  - 单例 + 缓存 + 任务队列；
///  - 通过 [com.github.kiastr.venera_ssr/colorize] MethodChannel 调用，
///    与原生 [ColorizeEngine.colorizeEsrgan] 对接（复用上色通道，无需新增原生方法）；
///  - 推理在原生后台线程执行，失败自动从 NNAPI 回退 CPU。
///
/// 与 v1（纯 Dart CPU 算法）并存：reader 侧按 `anime4KVersion` 选择引擎；
/// v4 仅 Android 生效（NNAPI/ONNX Runtime 为 Android 原生实现），非 Android 时
/// [isAvailable] 为 false，reader 自动回退 v1。
class Anime4KV4Service {
  Anime4KV4Service._internal();

  static final Anime4KV4Service _instance = Anime4KV4Service._internal();

  factory Anime4KV4Service() => _instance;

  static Anime4KV4Service get instance => _instance;

  /// 与原生端通信的 MethodChannel（复用上色通道）
  static const MethodChannel _channel =
      MethodChannel('com.github.kiastr.venera_ssr/colorize');

  String? _cacheDir;
  String? _modelPath;

  final Set<String> _processingKeys = {};
  static const int _maxConcurrentTasks = 2;
  int _runningTasks = 0;
  final List<Function> _taskQueue = [];

  /// 初始化缓存目录并探测模型（不自动下载）
  Future<void> init() async {
    try {
      final dir = await getTemporaryDirectory();
      _cacheDir = path.join(dir.path, 'anime4k_v4_cache');
      final cacheDirectory = Directory(_cacheDir!);
      if (!await cacheDirectory.exists()) {
        await cacheDirectory.create(recursive: true);
      }
      // 首次运行把打包进 APK 的模型抽取到应用目录（若已下载/自选或用户已删除则跳过）
      await Anime4KV4ModelManager.extractBundledModelIfNeeded();
      _modelPath = await Anime4KV4ModelManager.ensureModelAvailable();
    } catch (e) {
      Log.error('Anime4KV4', 'init error: $e');
    }
  }

  /// 模型是否可用（已下载到本地且当前平台支持）
  bool get isAvailable =>
      App.isAndroid && _modelPath != null;

  Future<bool> checkModelAvailable() async {
    if (_modelPath != null) {
      if (await File(_modelPath!).exists()) return true;
      _modelPath = null;
    }
    _modelPath = await Anime4KV4ModelManager.ensureModelAvailable();
    return _modelPath != null;
  }

  String? _getCachePath(String key) {
    if (_cacheDir == null) return null;
    return path.join(_cacheDir!, '${key.hashCode.abs()}.png');
  }

  Future<Uint8List?> _getFromCache(String key) async {
    final cachePath = _getCachePath(key);
    if (cachePath == null) return null;
    final file = File(cachePath);
    if (await file.exists()) {
      try {
        return await file.readAsBytes();
      } catch (e) {
        return null;
      }
    }
    return null;
  }

  Future<void> _saveToCache(String key, Uint8List data) async {
    final cachePath = _getCachePath(key);
    if (cachePath == null) return;
    try {
      final file = File(cachePath);
      await file.writeAsBytes(data);
    } catch (e) {
      Log.error('Anime4KV4', 'cache save error: $e');
    }
  }

  /// 调用原生端完成超分推理（type='esrgan'）。
  /// 优先 NNAPI（GPU），失败自动回退纯 CPU；任何失败返回 null（不抛异常）。
  Future<Uint8List?> _upscaleOnNative(
    Uint8List imageBytes,
    String modelPath,
    double intensity,
    bool useNnapi,
  ) async {
    try {
      final result = await _channel.invokeMethod<Uint8List>('colorize', {
        'imageBytes': imageBytes,
        'modelPath': modelPath,
        'type': 'esrgan',
        'useNnapi': useNnapi,
        'intensity': intensity,
      });
      return result;
    } catch (e, s) {
      Log.error('Anime4KV4', 'native upscale failed (useNnapi=$useNnapi): $e\n$s');
      return null;
    }
  }

  /// 丢弃原生端已缓存的 ONNX 会话（模型变更/导入/删除后必须调用）
  Future<void> resetNativeSession() async {
    try {
      await _channel.invokeMethod<void>('resetSession');
    } catch (e, s) {
      Log.error('Anime4KV4', 'resetNativeSession failed: $e\n$s');
    }
  }

  /// 处理图片字节数据，返回超分后的 PNG 字节数据（固定 4x）。
  ///
  /// 模型缺失或非 Android 时直接返回 null（上层据此回退 v1 或保持原图）。
  Future<Uint8List?> processImage({
    required Uint8List imageBytes,
    required String cacheKey,
    double intensity = 1.0,
  }) async {
    if (!App.isAndroid) {
      // v4 依赖 Android 原生 ONNX Runtime，非 Android 不处理（reader 自动回退 v1）
      return null;
    }
    if (_modelPath == null) {
      if (!await checkModelAvailable()) {
        Log.warning('Anime4KV4', 'Model not available, skipping for $cacheKey');
        return null;
      }
    }
    final modelPath = _modelPath;
    if (modelPath == null) return null;

    // v4 前缀 + intensity 避免与 v1 串图，且改强度后不返回旧缓存
    final fullKey = 'v4_${cacheKey}_${intensity.toStringAsFixed(2)}';

    final cached = await _getFromCache(fullKey);
    if (cached != null) {
      Log.info('Anime4KV4', 'cache hit for $cacheKey');
      return cached;
    }

    if (_processingKeys.contains(fullKey)) {
      Log.info('Anime4KV4', 'already processing $cacheKey');
      return null;
    }

    _processingKeys.add(fullKey);

    return _enqueueTask(() async {
      try {
        Log.info('Anime4KV4', 'processing image $cacheKey');

        var result = await _upscaleOnNative(imageBytes, modelPath, intensity, true);
        // NNAPI 失败（不支持/崩溃）时回退纯 CPU 重试一次
        if (result == null) {
          Log.warning('Anime4KV4', 'NNAPI failed, retry with CPU for $cacheKey');
          result = await _upscaleOnNative(imageBytes, modelPath, intensity, false);
        }

        if (result != null) {
          await _saveToCache(fullKey, result);
          Log.info('Anime4KV4', 'processing complete for $cacheKey');
        }
        return result;
      } catch (e, s) {
        Log.error('Anime4KV4', 'processing error: $e\n$s');
        return null;
      } finally {
        _processingKeys.remove(fullKey);
      }
    });
  }

  Future<T?> _enqueueTask<T>(Future<T?> Function() task) async {
    final completer = Completer<T?>();
    _taskQueue.add(() async {
      _runningTasks++;
      try {
        final result = await task();
        completer.complete(result);
      } catch (e) {
        completer.completeError(e);
      } finally {
        _runningTasks--;
        _nextTask();
      }
    });
    _nextTask();
    return completer.future;
  }

  void _nextTask() {
    if (_runningTasks < _maxConcurrentTasks && _taskQueue.isNotEmpty) {
      final task = _taskQueue.removeAt(0);
      task();
    }
  }

  Future<void> clearCache() async {
    if (_cacheDir == null) return;
    try {
      final dir = Directory(_cacheDir!);
      if (await dir.exists()) {
        await dir.delete(recursive: true);
        await dir.create(recursive: true);
      }
      Log.info('Anime4KV4', 'cache cleared');
    } catch (e) {
      Log.error('Anime4KV4', 'cache clear error: $e');
    }
  }

  Future<int> getCacheSize() async {
    if (_cacheDir == null) return 0;
    try {
      final dir = Directory(_cacheDir!);
      if (!await dir.exists()) return 0;
      int totalSize = 0;
      await for (final entity in dir.list(recursive: true)) {
        if (entity is File) {
          totalSize += await entity.length();
        }
      }
      return totalSize;
    } catch (e) {
      return 0;
    }
  }
}
