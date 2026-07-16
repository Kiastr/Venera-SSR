import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import 'package:venera/foundation/log.dart';

import 'colorization_processor.dart';

/// 图像上色服务
///
/// 基于 DeOldify ONNX 模型提供灰度图像上色功能，支持缓存机制以避免重复处理。
/// 采用单例模式，通过 [ColorizationService.instance] 访问。
///
/// 推理实现：原生（Kotlin + OpenCV + ONNX Runtime），通过 MethodChannel
/// [com.github.kiastr.venera_ssr/colorize] 调用。相比原纯 Dart `onnxruntime`
/// 实现，原生方案有会话缓存、正确的 OpenCV LAB 后处理（无 sRGB gamma 偏差）、
/// 正确的 NCHW 输出转置，且全程在原生后台线程执行，稳定性与速度都更优。
class ColorizationService {
  ColorizationService._internal();

  static final ColorizationService _instance = ColorizationService._internal();

  factory ColorizationService() => _instance;

  static ColorizationService get instance => _instance;

  /// 与原生端通信的 MethodChannel
  static const MethodChannel _channel =
      MethodChannel('com.github.kiastr.venera_ssr/colorize');

  /// 缓存目录路径
  String? _cacheDir;

  /// 模型文件路径（在应用首次启动时从 asset 拷贝到应用目录）
  String? _modelPath;

  /// 正在处理的任务集合（避免重复处理同一图片）
  final Set<String> _processingKeys = {};

  /// 最大并发处理数
  static const int _maxConcurrentTasks = 2;

  /// 当前正在运行的任务数
  int _runningTasks = 0;

  /// 任务队列
  final List<Function> _taskQueue = [];

  /// 初始化缓存目录，不自动下载模型
  Future<void> init() async {
    try {
      final dir = await getTemporaryDirectory();
      _cacheDir = path.join(dir.path, 'colorization_cache');
      final cacheDirectory = Directory(_cacheDir!);
      if (!await cacheDirectory.exists()) {
        await cacheDirectory.create(recursive: true);
      }
      // 检查模型是否已下载（不触发下载）
      _modelPath = await ColorizationModelManager.ensureModelAvailable();
    } catch (e) {
      Log.error('Colorization', 'Colorization init error: $e');
    }
  }

  /// 检查模型文件是否可用
  ///
  /// 会校验缓存的路径对应的文件是否仍然存在（模型可能已被删除），
  /// 不存在则重置并重新检测，确保删除模型后状态正确。
  Future<bool> checkModelAvailable() async {
    if (_modelPath != null) {
      if (await File(_modelPath!).exists()) return true;
      _modelPath = null; // 文件已被删除，重置
    }
    _modelPath = await ColorizationModelManager.ensureModelAvailable();
    return _modelPath != null;
  }

  /// 模型是否可用（已下载到本地）
  bool get isModelAvailable => _modelPath != null;

  /// 获取缓存文件路径
  String? _getCachePath(String key) {
    if (_cacheDir == null) return null;
    return path.join(_cacheDir!, '${key.hashCode.abs()}.png');
  }

  /// 检查是否有缓存，有则返回缓存数据
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

  /// 保存处理结果到缓存
  Future<void> _saveToCache(String key, Uint8List data) async {
    final cachePath = _getCachePath(key);
    if (cachePath == null) return;

    try {
      final file = File(cachePath);
      await file.writeAsBytes(data);
    } catch (e) {
      Log.error('Colorization', 'Colorization cache save error: $e');
    }
  }

  /// 调用原生端完成上色推理
  ///
  /// 返回上色后的 PNG 字节；失败返回 null（不抛异常，保持与上层调用契约一致）。
  Future<Uint8List?> _colorizeOnNative(
    Uint8List imageBytes,
    String modelPath,
    double intensity,
  ) async {
    try {
      final result = await _channel.invokeMethod<Uint8List>('colorize', {
        'imageBytes': imageBytes,
        'modelPath': modelPath,
        'type': 'deoldify',
        'useNnapi': false,
        'intensity': intensity,
      });
      return result;
    } catch (e, s) {
      Log.error('Colorization', 'native colorize failed: $e\n$s');
      return null;
    }
  }

  /// 通过原生 ContentResolver 把 content URI / 文件路径以有界分块（64KB）方式拷贝到 dest。
  ///
  /// 用于导入外部模型文件：避免把 ~243MB 模型一次性读入内存触发 OOM，
  /// 也避免 `XFile.openRead()` 在 content URI 下不可靠的流式实现把文件拷坏
  /// （这正是“选择外部模型崩溃”的根因——拷贝过程把整文件读入内存或产出损坏文件）。
  /// 返回写入的字节数。
  Future<int> copyUriTo(String uri, String dest) async {
    final r = await _channel.invokeMethod<int>('copyUri', {
      'uri': uri,
      'destPath': dest,
    });
    return r ?? 0;
  }

  /// 丢弃原生端已缓存的 ONNX 会话，使下次推理按当前模型路径重新加载。
  /// 导入/下载/删除模型后必须调用，否则 ModelManager 按路径缓存会沿用旧会话。
  Future<void> resetNativeSession() async {
    try {
      await _channel.invokeMethod<void>('resetSession');
    } catch (e, s) {
      Log.error('Colorization', 'resetNativeSession failed: $e\n$s');
    }
  }

  /// 处理图片字节数据，返回上色后的 PNG 字节数据
  ///
  /// [imageBytes] 原始图片字节数据
  /// [cacheKey] 缓存键，用于避免重复处理（通常使用图片 URL 或文件路径）
  /// [intensity] 上色强度（0.3 - 1.2，默认 1.0）
  Future<Uint8List?> processImage({
    required Uint8List imageBytes,
    required String cacheKey,
    double intensity = 1.0,
  }) async {
    // 模型文件还没准备好：兜底重新检测（init 时模型可能尚未下载）
    if (_modelPath == null) {
      if (!await checkModelAvailable()) {
        Log.warning('Colorization', 'Model not available, skipping colorization for $cacheKey');
        return null;
      }
    }
    // 用 final 捕获，确保在下方 _enqueueTask 闭包内能被提升为非空
    final modelPath = _modelPath;
    if (modelPath == null) {
      // 双保险：checkModelAvailable 返回 true 但 _modelPath 仍为 null（理论不应发生）
      return null;
    }

    // 生成唯一缓存键（包含参数信息）
    final fullKey = '${cacheKey}_${intensity.toStringAsFixed(2)}';

    // 优先从缓存读取
    final cached = await _getFromCache(fullKey);
    if (cached != null) {
      Log.info('Colorization', 'cache hit for $cacheKey');
      return cached;
    }

    // 防止重复处理同一图片
    if (_processingKeys.contains(fullKey)) {
      Log.info('Colorization', 'already processing $cacheKey');
      return null;
    }

    _processingKeys.add(fullKey);

    return _enqueueTask(() async {
      try {
        Log.info('Colorization', 'processing image $cacheKey, intensity: $intensity');

        final result = await _colorizeOnNative(imageBytes, modelPath, intensity);

        if (result != null) {
          // 保存到缓存
          await _saveToCache(fullKey, result);
          Log.info('Colorization', 'processing complete for $cacheKey');
        }

        return result;
      } catch (e, s) {
        Log.error('Colorization', 'Colorization processing error: $e\n$s');
        return null;
      } finally {
        _processingKeys.remove(fullKey);
      }
    });
  }

  /// 将任务加入队列并按序执行
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

  /// 执行下一个任务
  void _nextTask() {
    if (_runningTasks < _maxConcurrentTasks && _taskQueue.isNotEmpty) {
      final task = _taskQueue.removeAt(0);
      task();
    }
  }

  /// 处理本地图片文件
  Future<Uint8List?> processFile({
    required String filePath,
    double intensity = 1.0,
  }) async {
    try {
      final file = File(filePath);
      if (!await file.exists()) return null;

      final imageBytes = await file.readAsBytes();
      return processImage(
        imageBytes: imageBytes,
        cacheKey: filePath,
        intensity: intensity,
      );
    } catch (e) {
      Log.error('Colorization', 'Colorization file processing error: $e');
      return null;
    }
  }

  /// 清除所有上色缓存
  Future<void> clearCache() async {
    if (_cacheDir == null) return;
    try {
      final dir = Directory(_cacheDir!);
      if (await dir.exists()) {
        await dir.delete(recursive: true);
        await dir.create(recursive: true);
      }
      Log.info('Colorization', 'Colorization: cache cleared');
    } catch (e) {
      Log.error('Colorization', 'Colorization cache clear error: $e');
    }
  }

  /// 获取缓存占用的磁盘大小（字节）
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
