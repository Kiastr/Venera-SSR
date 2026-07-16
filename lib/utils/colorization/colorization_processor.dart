import 'dart:io';

import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:venera/foundation/log.dart';

/// 模型管理器：负责从网络下载 DeOldify / DDColor 模型到应用目录。
///
/// 不使用 Flutter assets 打包模型（模型文件 ~243MB 会导致 Android APK 构建失败），
/// 改为在设置页由用户手动触发下载，避免首次启动即下载大文件。
///
/// 实际的推理已移至原生端（Kotlin + OpenCV + ONNX Runtime），
/// 见 android/app/src/main/kotlin/com/github/kiastr/venera_ssr/colorize/，
/// 通过 MethodChannel 调用。本文件只负责模型的下载与本地路径解析。
///
/// 模型调用位置（Model Invocation Location）：
///   [getApplicationSupportDirectory]/deoldify_artistic.onnx
/// 即 ColorizationService 通过 [ensureModelAvailable] 取得、原生 ColorizeEngine 经
/// createSession(modelPath) 直接读取的路径。无论是下载模型、int8 变体还是用户自选的
/// 外部模型，最终都落盘到这一位置——彻底消除“自定义路径 + shared_prefs 间接层”带来的
/// 异常与崩溃。
class ColorizationModelManager {
  static const modelFileName = 'deoldify_artistic.onnx';
  static const expectedFileSize = 243 * 1024 * 1024; // ~243MB（DeOldify 完整版）

  /// 判定一个 onnx 是否为有效模型的最小体积（8MB）。
  /// DeOldify 完整版 ~243MB，DDColor int8 轻量版 ~60MB，二者都满足；
  /// 用统一下限避免把 int8 轻量模型误判为“未下载”。
  static const int _validModelMinSize = 8 * 1024 * 1024;

  /// 默认镜像源（DeOldify，按稳定性排序），用户可在设置页增删
  static const String _modelUrlsKey = 'colorization_model_urls';

  /// 是否正在使用“自选外部模型”（落盘到模型调用位置，覆盖下载模型）
  static const String _customModelActiveKey = 'colorization_custom_model_active';

  /// 自选外部模型的原始文件名（仅用于 UI 展示）
  static const String _customModelNameKey = 'colorization_custom_model_name';

  /// 当前选中的模型变体：'deoldify' | 'deoldify-int8'
  static const String _variantKey = 'colorization_model_variant';

  static const List<String> _defaultModelUrls = [
    'https://mirror.ghproxy.com/https://github.com/instant-high/deoldify-onnx/releases/download/deoldify-onnx/deoldify.onnx',
    'https://ghp.ci/https://github.com/instant-high/deoldify-onnx/releases/download/deoldify-onnx/deoldify.onnx',
    'https://ghproxy.net/https://github.com/instant-high/deoldify-onnx/releases/download/deoldify-onnx/deoldify.onnx',
    'https://github.com/instant-high/deoldify-onnx/releases/download/deoldify-onnx/deoldify.onnx',
  ];

  /// DeOldify int8 轻量版镜像（实验性功能，用户实测可直接复用现有推理逻辑，无需改动）
  static const List<String> _int8ModelUrls = [
    'https://ghproxy.net/https://github.com/Kiastr/AiColorize/releases/download/models/deoldify_int8.onnx',
    'https://github.com/Kiastr/AiColorize/releases/download/models/deoldify_int8.onnx',
  ];

  /// 可选模型变体（设置页切换下载源；推理逻辑不变）
  static const List<ColorizationModelVariant> modelVariants = [
    ColorizationModelVariant('deoldify', 'DeOldify Artistic'),
    ColorizationModelVariant('deoldify-int8', 'DeOldify int8 (轻量, 实验性)'),
  ];

  /// 持久化的镜像 URL 列表（用户可编辑）；首次运行初始化为默认列表
  static List<String> _modelUrls = [];
  static bool _urlsLoaded = false;

  /// 模型调用位置的文件是否已因“自选外部模型”被覆盖
  static bool _customModelActive = false;

  /// 自选外部模型原始文件名
  static String? _customModelName;

  /// 当前选中的变体
  static String _selectedVariant = 'deoldify';

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

  /// 添加一个自定义镜像 URL（去重）
  static Future<void> addModelUrl(String url) async {
    await getModelUrls();
    final trimmed = url.trim();
    if (trimmed.isEmpty || _modelUrls.contains(trimmed)) return;
    _modelUrls.add(trimmed);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList(_modelUrlsKey, _modelUrls);
  }

  /// 删除指定下标的镜像 URL
  static Future<void> removeModelUrlAt(int index) async {
    await getModelUrls();
    if (index < 0 || index >= _modelUrls.length) return;
    _modelUrls.removeAt(index);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList(_modelUrlsKey, _modelUrls);
  }

  /// 恢复默认镜像列表
  static Future<void> resetModelUrls() async {
    _modelUrls = List.from(_defaultModelUrls);
    _urlsLoaded = true;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList(_modelUrlsKey, _modelUrls);
  }

  /// 是否正在使用自选外部模型
  static Future<bool> isCustomModelActive() async {
    if (_customModelActive) return true;
    final prefs = await SharedPreferences.getInstance();
    _customModelActive = prefs.getBool(_customModelActiveKey) ?? false;
    return _customModelActive;
  }

  /// 自选外部模型的原始文件名（无则返回 null）
  static Future<String?> getCustomModelName() async {
    if (_customModelName != null) return _customModelName;
    final prefs = await SharedPreferences.getInstance();
    _customModelName = prefs.getString(_customModelNameKey);
    return _customModelName;
  }

  /// 回退到内置（下载）模型：删除被覆盖的模型调用位置文件，
  /// 若存在此前备份的下载模型则还原。
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
  /// 实际文件拷贝由调用方经原生 [ColorizationService.copyUriTo] 通道完成，
  /// 本方法只负责记账，使 UI/服务能识别当前处于自选模型状态。
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

  /// 有效模型的最小体积（字节），供 UI 侧拷贝后做体积校验
  static int get validModelMinSize => _validModelMinSize;

  static String? _cachedModelPath;
  static bool _isDownloading = false;
  static double _downloadProgress = 0.0;
  static String? _currentStatus;

  /// 获取模型文件路径（即模型调用位置）。
  /// 不自动下载；文件不存在或无效则返回 null。
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
      // 文件不完整，删除
      await targetFile.delete();
    }

    return null;
  }

  /// 当前选中的模型变体
  static Future<String> getSelectedVariant() async {
    final prefs = await SharedPreferences.getInstance();
    _selectedVariant = prefs.getString(_variantKey) ?? 'deoldify';
    return _selectedVariant;
  }

  /// 设置选中的模型变体（影响下载源）
  static Future<void> setSelectedVariant(String id) async {
    _selectedVariant = id;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_variantKey, id);
  }

  /// 取指定变体的下载 URL 列表（含镜像回退）
  static List<String> _urlsForVariant(String variant) {
    if (variant == 'deoldify-int8') return List.from(_int8ModelUrls);
    return List.from(_defaultModelUrls); // 用户编辑的镜像列表在 downloadModel 内取
  }

  /// 手动触发模型下载，支持进度回调和断点续传。
  /// [variant] 指定下载哪个变体（'deoldify' 默认，'deoldify-int8' 轻量）。
  static Future<void> downloadModel({
    String variant = 'deoldify',
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
      // 删除旧的临时文件
      if (await tempFile.exists()) {
        await tempFile.delete();
      }

      Exception? lastError;
      // deoldify 用用户可编辑的镜像列表；int8 用内置镜像
      final urls =
          variant == 'deoldify-int8'
              ? _urlsForVariant(variant)
              : await getModelUrls();
      final minSize =
          variant == 'deoldify-int8'
              ? _validModelMinSize
              : expectedFileSize ~/ 2;

      for (int i = 0; i < urls.length; i++) {
        final url = urls[i];
        reportStatus('Trying mirror ${i + 1}/${urls.length}...');
        try {
          await _downloadWithResume(url, tempPath, (received, total) {
            _downloadProgress = total > 0 ? received / total : 0;
            onProgress?.call(_downloadProgress);
          });
          final downloaded = await File(tempPath).length();
          if (downloaded > minSize) {
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
          // 清理临时文件
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

  /// 使用 HttpClient 下载，支持 Range 断点续传和实时进度
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

  /// 模型是否已下载且完整（含自选外部模型与已落盘到调用位置的任意变体）
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

  /// 获取已下载模型大小（字节）
  static Future<int> getDownloadedSize() async {
    final dir = await getApplicationSupportDirectory();
    final targetPath = path.join(dir.path, modelFileName);
    final targetFile = File(targetPath);
    if (!await targetFile.exists()) return 0;
    return await targetFile.length();
  }

  /// 获取下载进度（0.0 - 1.0）
  static double get downloadProgress => _downloadProgress;

  /// 是否正在下载
  static bool get isDownloading => _isDownloading;

  /// 当前状态文本
  static String? get currentStatus => _currentStatus;

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
}

/// 模型变体描述（设置页用于切换下载源；推理逻辑不变）
class ColorizationModelVariant {
  final String id;
  final String label;

  const ColorizationModelVariant(this.id, this.label);
}
