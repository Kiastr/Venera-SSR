import 'dart:io';

import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:venera/foundation/log.dart';
import 'package:venera/utils/colorization/colorization_service.dart';

/// v4 超分模型管理器：管理 Real-ESRGAN(animevideov3) ONNX 模型（~4MB）的生命周期。
///
/// 模型获取策略（三选一，优先级从高到低）：
///  1. 自选外部模型（用户从本地导入，最高优先，绝不被覆盖）；
///  2. 打包进 APK 的内置模型（首次运行经 [extractBundledModelIfNeeded] 抽取到应用目录，
///     开箱即用、无需联网）；
///  3. 运行时下载（下载管理器保留：用户删除内置模型后可重新下载，或切换镜像/自选模型）。
///
/// 其他约定：
///  - 通过 [ColorizationService] 复用的 [com.github.kiastr.venera_ssr/colorize] MethodChannel
///    的 `copyUri` 方法完成“自选本地模型”的拷贝（不额外新增原生方法）。
///  - 模型调用位置固定为 [getApplicationSupportDirectory]/realesr_animevideov3.onnx，
///    原生 [ColorizeEngine.colorizeEsrgan] 经 createSession(modelPath) 直接读取。
class Anime4KV4ModelManager {
  static const modelFileName = 'realesr_animevideov3.onnx';

  /// 有效模型最小体积（2MB）。animevideov3 约 4MB，含此下限避免把损坏/空文件当有效模型。
  static const int _validModelMinSize = 2 * 1024 * 1024;

  static const String _modelUrlsKey = 'anime4kV4_model_urls';

  /// 是否正在使用“自选外部模型”（覆盖下载模型）
  static const String _customModelActiveKey = 'anime4kV4_custom_model_active';

  /// 自选外部模型的原始文件名（仅用于 UI 展示）
  static const String _customModelNameKey = 'anime4kV4_custom_model_name';

  /// 打包进 APK 的模型在 assets 中的路径（需同步在 pubspec.yaml 的 assets: 中声明）。
  static const String _bundledAssetPath = 'assets/models/realesr_animevideov3.onnx';

  /// 标记“打包模型已抽取到应用目录”，确保仅抽取一次；用户手动删除后不自动回灌。
  static const String _bundledInstalledKey = 'anime4kV4_bundled_installed';

  static const List<String> _defaultModelUrls = [
    'https://ghproxy.net/https://github.com/Kiastr/Venera-SSR/releases/download/model/realesr_animevideov3.onnx',
    'https://github.com/Kiastr/Venera-SSR/releases/download/model/realesr_animevideov3.onnx',
  ];

  static List<String> _modelUrls = [];
  static bool _urlsLoaded = false;

  static bool _customModelActive = false;
  static String? _customModelName;

  static String? _cachedModelPath;
  static bool _isDownloading = false;
  static double _downloadProgress = 0.0;
  static String? _currentStatus;

  /// 有效模型的最小体积（字节），供 UI 侧拷贝后做体积校验
  static int get validModelMinSize => _validModelMinSize;

  /// 获取当前生效的镜像 URL 列表（懒加载 + 持久化）
  static Future<List<String>> getModelUrls() async {
    if (!_urlsLoaded) {
      final prefs = await SharedPreferences.getInstance();
      final saved = prefs.getStringList(_modelUrlsKey);
      _modelUrls =
          (saved != null && saved.isNotEmpty)
              ? List.from(saved)
              : List.from(_defaultModelUrls);
      _urlsLoaded = true;
    }
    return List.from(_modelUrls);
  }

  static Future<void> addModelUrl(String url) async {
    await getModelUrls();
    final trimmed = url.trim();
    if (trimmed.isEmpty || _modelUrls.contains(trimmed)) return;
    _modelUrls.add(trimmed);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList(_modelUrlsKey, _modelUrls);
  }

  static Future<void> removeModelUrlAt(int index) async {
    await getModelUrls();
    if (index < 0 || index >= _modelUrls.length) return;
    _modelUrls.removeAt(index);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList(_modelUrlsKey, _modelUrls);
  }

  static Future<void> resetModelUrls() async {
    _modelUrls = List.from(_defaultModelUrls);
    _urlsLoaded = true;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList(_modelUrlsKey, _modelUrls);
  }

  static Future<bool> isCustomModelActive() async {
    if (_customModelActive) return true;
    final prefs = await SharedPreferences.getInstance();
    _customModelActive = prefs.getBool(_customModelActiveKey) ?? false;
    return _customModelActive;
  }

  static Future<String?> getCustomModelName() async {
    if (_customModelName != null) return _customModelName;
    final prefs = await SharedPreferences.getInstance();
    _customModelName = prefs.getString(_customModelNameKey);
    return _customModelName;
  }

  /// 回退到内置（下载）模型：删除被覆盖的模型调用位置文件，若存在此前备份则还原。
  static Future<void> clearCustomModelSelection() async {
    final dir = await getApplicationSupportDirectory();
    final targetPath = path.join(dir.path, modelFileName);
    final bakPath = '$targetPath.bak';
    final targetFile = File(targetPath);
    if (await targetFile.exists()) {
      await targetFile.delete();
    }
    if (await File(bakPath).exists()) {
      await File(bakPath).rename(targetPath);
    }
    _cachedModelPath = null;
    _customModelActive = false;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_customModelActiveKey, false);
    await prefs.remove(_customModelNameKey);
  }

  /// 标记“模型调用位置的文件”为自选外部模型（写 prefs + 刷新缓存路径）。
  static Future<void> markCustomModelActive(String displayName) async {
    final dir = await getApplicationSupportDirectory();
    final targetPath = path.join(dir.path, modelFileName);
    _cachedModelPath = targetPath;
    _customModelActive = true;
    _customModelName = displayName;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_customModelActiveKey, true);
    await prefs.setString(_customModelNameKey, displayName);
  }

  /// 把打包进 APK 的模型（assets）抽取到应用目录（模型调用位置），仅在需要时执行一次。
  ///
  /// 触发条件（全部满足才回灌）：
  ///  1. 未启用“自选外部模型”（用户自选优先，绝不覆盖）；
  ///  2. 模型调用位置当前没有有效文件（不存在或体积 < 下限）；
  ///  3. 尚未标记为“打包模型已安装”——用户手动删除后不自动回灌，尊重用户意愿。
  ///
  /// assets 中缺少该模型（例如尚未打包）时静默跳过，让下载管理器接管，保证 APK 始终可构建。
  static Future<void> extractBundledModelIfNeeded() async {
    try {
      // 用户自选外部模型时不覆盖
      if (await isCustomModelActive()) return;

      final prefs = await SharedPreferences.getInstance();
      if (prefs.getBool(_bundledInstalledKey) ?? false) {
        // 已抽取过一次；若用户随后手动删除，不再自动回灌
        return;
      }

      final dir = await getApplicationSupportDirectory();
      final targetPath = path.join(dir.path, modelFileName);
      final targetFile = File(targetPath);

      // 已有有效模型（下载/导入）则无需抽取，直接标记完成
      if (await targetFile.exists() &&
          await targetFile.length() > _validModelMinSize) {
        await prefs.setBool(_bundledInstalledKey, true);
        return;
      }

      // 从 assets 读取打包模型；不存在（未打包）则静默跳过
      final ByteData data;
      try {
        data = await rootBundle.load(_bundledAssetPath);
      } catch (_) {
        // assets 未包含该模型：交给下载管理器，保持可构建/可降级
        return;
      }

      final bytes = data.buffer.asUint8List(
        data.offsetInBytes,
        data.lengthInBytes,
      );
      if (bytes.length <= _validModelMinSize) {
        // 打包文件异常（过小），不写入，交给下载流程
        return;
      }

      // 原子写入：先写临时文件再重命名，避免中断产生半截文件
      final tmpPath = '$targetPath.bundle.tmp';
      final tmpFile = File(tmpPath);
      await tmpFile.writeAsBytes(bytes, flush: true);
      await tmpFile.rename(targetPath);

      await prefs.setBool(_bundledInstalledKey, true);
      _cachedModelPath = targetPath;
      Log.info('Anime4KV4',
          'bundled model extracted to $targetPath (${bytes.length} bytes)');
    } catch (e, s) {
      Log.error('Anime4KV4', 'extractBundledModelIfNeeded failed: $e\n$s');
    }
  }

  /// 获取模型文件路径（即模型调用位置）。不自动下载；文件不存在或无效则返回 null。
  static Future<String?> ensureModelAvailable() async {
    if (_cachedModelPath != null) {
      final f = File(_cachedModelPath!);
      if (await f.exists() && await f.length() > _validModelMinSize) {
        return _cachedModelPath!;
      }
      _cachedModelPath = null;
    }

    final dir = await getApplicationSupportDirectory();
    final targetPath = path.join(dir.path, modelFileName);
    final targetFile = File(targetPath);

    if (await targetFile.exists()) {
      final size = await targetFile.length();
      if (size > _validModelMinSize) {
        _cachedModelPath = targetPath;
        return targetPath;
      }
      await targetFile.delete();
    }

    return null;
  }

  /// 模型是否已下载且完整（含自选外部模型）
  static Future<bool> get isModelDownloaded async {
    final dir = await getApplicationSupportDirectory();
    final targetPath = path.join(dir.path, modelFileName);
    final targetFile = File(targetPath);
    if (await targetFile.exists()) {
      final size = await targetFile.length();
      if (size > _validModelMinSize) return true;
    }
    if (_cachedModelPath != null) return true;
    return false;
  }

  static Future<int> getDownloadedSize() async {
    final dir = await getApplicationSupportDirectory();
    final targetPath = path.join(dir.path, modelFileName);
    final targetFile = File(targetPath);
    if (!await targetFile.exists()) return 0;
    return await targetFile.length();
  }

  static double get downloadProgress => _downloadProgress;
  static bool get isDownloading => _isDownloading;
  static String? get currentStatus => _currentStatus;

  /// 手动触发模型下载，支持进度回调和断点续传。
  static Future<void> downloadModel({
    void Function(double progress)? onProgress,
    void Function(String status)? onStatus,
  }) async {
    if (_isDownloading) {
      throw Exception('Model is already downloading');
    }

    final dir = await getApplicationSupportDirectory();
    final targetPath = path.join(dir.path, modelFileName);
    final tempPath = '$targetPath.tmp';
    final tempFile = File(tempPath);

    _isDownloading = true;
    _downloadProgress = 0.0;

    void reportStatus(String status) {
      _currentStatus = status;
      onStatus?.call(status);
    }

    try {
      if (await tempFile.exists()) {
        await tempFile.delete();
      }

      Exception? lastError;
      final urls = await getModelUrls();
      for (int i = 0; i < urls.length; i++) {
        final url = urls[i];
        reportStatus('Trying mirror ${i + 1}/${urls.length}...');
        try {
          await _downloadWithResume(url, tempPath, (received, total) {
            _downloadProgress = total > 0 ? received / total : 0;
            onProgress?.call(_downloadProgress);
          });
          final downloaded = await File(tempPath).length();
          if (downloaded > _validModelMinSize) {
            await File(tempPath).rename(targetPath);
            _cachedModelPath = targetPath;
            _customModelActive = false;
            final prefs = await SharedPreferences.getInstance();
            await prefs.setBool(_customModelActiveKey, false);
            await prefs.remove(_customModelNameKey);
            _downloadProgress = 1.0;
            reportStatus('Download complete');
            return;
          }
          await File(tempPath).delete();
        } catch (e) {
          lastError = e is Exception ? e : Exception(e.toString());
          reportStatus('Mirror ${i + 1} failed: $e');
          if (await tempFile.exists()) {
            await tempFile.delete();
          }
        }
      }
      throw lastError ?? Exception('All model download URLs failed');
    } finally {
      _isDownloading = false;
    }
  }

  static Future<void> _downloadWithResume(
    String url,
    String targetPath,
    void Function(int received, int total) onProgress,
  ) async {
    final client = HttpClient();
    final file = File(targetPath);
    int startByte = 0;
    if (await file.exists()) {
      startByte = await file.length();
    }

    IOSink? sink;
    try {
      final uri = Uri.parse(url);
      final request = await client.getUrl(uri);
      request.followRedirects = true;
      request.headers.set('User-Agent', 'Venera/1.0');
      request.headers.set('Accept', '*/*');
      request.headers.set('Connection', 'keep-alive');
      if (startByte > 0) {
        request.headers.set('Range', 'bytes=$startByte-');
      }

      final response = await request.close();
      if (response.statusCode != HttpStatus.ok &&
          response.statusCode != HttpStatus.partialContent) {
        throw Exception('HTTP ${response.statusCode}');
      }

      final contentLength = response.contentLength;
      int received = startByte;
      final total = contentLength > 0 ? contentLength + startByte : 0;

      sink = file.openWrite(
        mode: startByte > 0 ? FileMode.append : FileMode.write,
      );

      await for (final chunk in response) {
        sink.add(chunk);
        received += chunk.length;
        onProgress(received, total);
      }
      await sink.close();
    } catch (e) {
      sink?.close();
      rethrow;
    } finally {
      client.close();
    }
  }

  /// 清除模型文件（含自选外部模型备份）
  static Future<void> clearModel() async {
    final dir = await getApplicationSupportDirectory();
    final targetPath = path.join(dir.path, modelFileName);
    final tempPath = '$targetPath.tmp';
    final bakPath = '$targetPath.bak';
    final targetFile = File(targetPath);
    final tempFile = File(tempPath);
    final bakFile = File(bakPath);
    if (await targetFile.exists()) {
      await targetFile.delete();
    }
    if (await tempFile.exists()) {
      await tempFile.delete();
    }
    if (await bakFile.exists()) {
      await bakFile.delete();
    }
    _cachedModelPath = null;
    _customModelActive = false;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_customModelActiveKey, false);
    await prefs.remove(_customModelNameKey);
  }

  /// 通过原生 ContentResolver（colorize 通道的 copyUri）把 content URI / 文件路径以
  /// 有界分块（64KB）方式拷贝到模型调用位置，避免一次性读入内存（OOM）或拷坏。
  /// 实际文件拷贝由 [ColorizationService.copyUriTo] 经原生完成。
  static Future<int> copyUriTo(String uri, String dest) =>
      ColorizationService.instance.copyUriTo(uri, dest);
}
