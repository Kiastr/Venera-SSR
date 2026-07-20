import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:venera/foundation/log.dart';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;

import 'anime4k_upscaler.dart';

/// Anime4K 超分服务
///
/// 提供图像超分辨率处理功能，支持缓存机制以避免重复处理。
/// 采用单例模式，通过 [Anime4KService.instance] 访问。
class Anime4KService {
  Anime4KService._internal();

  static final Anime4KService _instance = Anime4KService._internal();

  factory Anime4KService() => _instance;

  static Anime4KService get instance => _instance;

  /// 缓存目录路径
  String? _cacheDir;

  /// 正在处理的任务集合（避免重复处理同一图片）
  final Set<String> _processingKeys = {};

  /// 最大并发处理数
  static const int _maxConcurrentTasks = 2;

  /// 当前正在运行的任务数
  int _runningTasks = 0;

  /// 任务队列
  final List<Function> _taskQueue = [];

  /// 初始化缓存目录
  Future<void> init() async {
    try {
      final dir = await getTemporaryDirectory();
      _cacheDir = path.join(dir.path, 'anime4k_cache');
      final cacheDirectory = Directory(_cacheDir!);
      if (!await cacheDirectory.exists()) {
        await cacheDirectory.create(recursive: true);
      }
    } catch (e) {
      Log.error('Anime4K', 'Anime4K cache init error: $e');
    }
  }

  /// 获取缓存文件路径
  ///
  /// 使用 key 的 SHA-256 作为文件名，避免 [String.hashCode] 哈希碰撞导致
  /// 不同图片命中彼此的缓存（显示错图）。
  String? _getCachePath(String key) {
    if (_cacheDir == null) return null;
    final hash = sha256.convert(utf8.encode(key)).toString();
    return path.join(_cacheDir!, '$hash.png');
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
      Log.error('Anime4K', 'Anime4K cache save error: $e');
    }
  }

  /// 处理图片字节数据，返回超分后的 PNG 字节数据
  ///
  /// [imageBytes] 原始图片字节数据
  /// [cacheKey] 缓存键，用于避免重复处理（通常使用图片 URL 或文件路径）
  /// [scaleFactor] 放大倍数（默认 2.0）
  /// [pushStrength] 线条细化强度（默认 0.31）
  /// [pushGradStrength] 梯度精炼强度（默认 1.0）
  Future<Uint8List?> processImage({
    required Uint8List imageBytes,
    required String cacheKey,
    double scaleFactor = 2.0,
    double pushStrength = 0.31,
    double pushGradStrength = 1.0,
  }) async {
    // 生成唯一缓存键（包含参数信息）
    final fullKey =
        '${cacheKey}_${scaleFactor}_${pushStrength}_$pushGradStrength';

    // 优先从缓存读取
    final cached = await _getFromCache(fullKey);
    if (cached != null) {
      Log.info('Anime4K', 'cache hit for $cacheKey');
      return cached;
    }

    // 防止重复处理同一图片
    if (_processingKeys.contains(fullKey)) {
      Log.info('Anime4K', 'already processing $cacheKey');
      return null;
    }

    _processingKeys.add(fullKey);

    return _enqueueTask(() async {
      try {
        Log.info('Anime4K', 'processing image $cacheKey, '
            'scale: $scaleFactor, push: $pushStrength, grad: $pushGradStrength');

        final params = Anime4KParams(
          imageBytes: imageBytes,
          pushStrength: pushStrength,
          pushGradStrength: pushGradStrength,
          scaleFactor: scaleFactor,
        );

        final result = await Anime4KUpscaler.processInIsolate(params);

        if (result != null) {
          // 保存到缓存
          await _saveToCache(fullKey, result);
          Log.info('Anime4K', 'processing complete for $cacheKey');
        }

        return result;
      } catch (e) {
        Log.error('Anime4K', 'Anime4K processing error: $e');
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
  ///
  /// [filePath] 本地图片文件路径
  Future<Uint8List?> processFile({
    required String filePath,
    double scaleFactor = 2.0,
    double pushStrength = 0.31,
    double pushGradStrength = 1.0,
  }) async {
    try {
      final file = File(filePath);
      if (!await file.exists()) return null;

      final imageBytes = await file.readAsBytes();
      return processImage(
        imageBytes: imageBytes,
        cacheKey: filePath,
        scaleFactor: scaleFactor,
        pushStrength: pushStrength,
        pushGradStrength: pushGradStrength,
      );
    } catch (e) {
      Log.error('Anime4K', 'Anime4K file processing error: $e');
      return null;
    }
  }

  /// 清除所有超分缓存
  Future<void> clearCache() async {
    if (_cacheDir == null) return;
    try {
      final dir = Directory(_cacheDir!);
      if (await dir.exists()) {
        await dir.delete(recursive: true);
        await dir.create(recursive: true);
      }
      Log.info('Anime4K', 'Anime4K: cache cleared');
    } catch (e) {
      Log.error('Anime4K', 'Anime4K cache clear error: $e');
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
