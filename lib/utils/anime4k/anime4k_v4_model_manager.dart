import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:venera/foundation/log.dart';
import 'package:venera/utils/colorization/colorization_service.dart';

/// 单个 v4 超分模型的定义。
///
/// [scale] 仅作 UI 提示；真正的放大倍数由原生 [ColorizeEngine.getScale] 从模型
/// 实际输入/输出维度探测，因此换权重（4x/2x）无需改原生代码。
class V4ModelDef {
  final String id;
  final String fileName;
  final String displayName;
  final int scale;
  final int sizeHintMB;
  final String? bundledAssetPath;
  final List<String> defaultUrls;

  const V4ModelDef({
    required this.id,
    required this.fileName,
    required this.displayName,
    required this.scale,
    required this.sizeHintMB,
    this.bundledAssetPath,
    required this.defaultUrls,
  });
}

/// v4 超分模型管理器：管理 Real-ESRGAN ONNX 模型（默认 4x animevideov3，可选 2x general-x2c）
/// 的生命周期。支持多模型，所有对外方法按“当前选中模型”路由，调用方无需传 modelId。
///
/// 模型获取策略（三选一，优先级从高到低）：
///  1. 自选外部模型（用户从本地导入，最高优先，绝不被覆盖）；
///  2. 打包进 APK 的内置模型（首次运行经 [extractBundledModelIfNeeded] 抽取到应用目录，
///     开箱即用、无需联网；2x 模型默认不打包，走下载）；
///  3. 运行时下载（下载管理器保留：用户删除内置模型后可重新下载，或切换镜像/自选模型）。
///
/// 其他约定：
///  - 通过 [ColorizationService] 复用的 [com.github.kiastr.venera_ssr/colorize] MethodChannel
///    的 `copyUri` 方法完成“自选本地模型”的拷贝（不额外新增原生方法）。
///  - 每个模型的调用位置为 [getApplicationSupportDirectory]/<fileName>，
///    原生 [ColorizeEngine.colorizeEsrgan] 经 createSession(modelPath) 直接读取。
class Anime4KV4ModelManager {
  /// 模型注册表：4x 动画模型 + 2x 通用模型。新增权重只需在此追加一项。
  static final List<V4ModelDef> models = [
    V4ModelDef(
      id: 'anime4k_x4',
      fileName: 'realesr_animevideov3.onnx',
      displayName: '动画 4× (animevideov3)',
      scale: 4,
      sizeHintMB: 4,
      bundledAssetPath: 'assets/models/realesr_animevideov3.onnx',
      defaultUrls: [
        'https://ghproxy.net/https://github.com/Kiastr/Venera-SSR/releases/download/model/realesr_animevideov3.onnx',
        'https://github.com/Kiastr/Venera-SSR/releases/download/model/realesr_animevideov3.onnx',
      ],
    ),
    V4ModelDef(
      id: 'general_x2',
      fileName: 'realesr_general_x2c.onnx',
      displayName: '通用 2× (general-x2c)',
      scale: 2,
      sizeHintMB: 8,
      bundledAssetPath: null, // 2x 默认不打包，运行时下载
      defaultUrls: [
        'https://ghproxy.net/https://github.com/Kiastr/Venera-SSR/releases/download/model/realesr_general_x2c.onnx',
        'https://github.com/Kiastr/Venera-SSR/releases/download/model/realesr_general_x2c.onnx',
      ],
    ),
  ];

  /// 有效模型最小体积（2MB）。animevideov3 约 4MB、general-x2c 约 8MB，含此下限避免
  /// 把损坏/空文件当有效模型。
  static const int _validModelMinSize = 2 * 1024 * 1024;

  static const String _selectedModelKey = 'anime4kV4_selected_model';
  static String _selectedModelId = 'anime4k_x4';

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

  static V4ModelDef _modelById(String id) =>
      models.firstWhere((m) => m.id == id, orElse: () => models.first);

  /// 当前选中模型的定义
  static V4ModelDef get selectedDef => _modelById(_selectedModelId);

  /// 当前选中模型的文件名（调用位置文件名）
  static String get modelFileName => selectedDef.fileName;

  /// 全部可用模型（供 UI 构建选择器）
  static List<V4ModelDef> getModels() => List.unmodifiable(models);

  static bool isValidModelId(String id) => models.any((m) => m.id == id);

  /// 切换当前选中模型并持久化；重置内存态（含下载/自选状态），下次读取从 prefs 重载。
  static Future<void> setSelectedModelId(String id) async {
    if (!isValidModelId(id)) return;
    _selectedModelId = id;
    _resetMemState();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_selectedModelKey, id);
  }

  static void _resetMemState() {
    _urlsLoaded = false;
    _customModelActive = false;
    _customModelName = null;
    _cachedModelPath = null;
    _isDownloading = false;
    _downloadProgress = 0.0;
    _currentStatus = null;
  }

  /// 获取当前生效的镜像 URL 列表（懒加载 + 持久化）
  static Future<List<String>> getModelUrls() async {
    final def = selectedDef;
    final key = 'anime4kV4_urls_${def.id}';
    if (!_urlsLoaded) {
      final prefs = await SharedPreferences.getInstance();
      final saved = prefs.getStringList(key);
      _modelUrls = (saved != null && saved.isNotEmpty)
          ? List.from(saved)
          : List.from(def.defaultUrls);
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
    await prefs.setStringList('anime4kV4_urls_${selectedDef.id}', _modelUrls);
  }

  static Future<void> removeModelUrlAt(int index) async {
    await getModelUrls();
    if (index < 0 || index >= _modelUrls.length) return;
    _modelUrls.removeAt(index);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList('anime4kV4_urls_${selectedDef.id}', _modelUrls);
  }

  static Future<void> resetModelUrls() async {
    final def = selectedDef;
    _modelUrls = List.from(def.defaultUrls);
    _urlsLoaded = true;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList('anime4kV4_urls_${def.id}', _modelUrls);
  }

  static Future<bool> isCustomModelActive() async {
    if (_customModelActive) return true;
    final def = selectedDef;
    final prefs = await SharedPreferences.getInstance();
    _customModelActive =
        prefs.getBool('anime4kV4_custom_${def.id}') ?? false;
    return _customModelActive;
  }

  static Future<String?> getCustomModelName() async {
    if (_customModelName != null) return _customModelName;
    final def = selectedDef;
    final prefs = await SharedPreferences.getInstance();
    _customModelName = prefs.getString('anime4kV4_custom_name_${def.id}');
    return _customModelName;
  }

  /// 回退到内置（下载）模型：删除被覆盖的模型调用位置文件，若存在此前备份则还原。
  static Future<void> clearCustomModelSelection() async {
    final def = selectedDef;
    final dir = await getApplicationSupportDirectory();
    final targetPath = path.join(dir.path, def.fileName);
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
    await prefs.setBool('anime4kV4_custom_${def.id}', false);
    await prefs.remove('anime4kV4_custom_name_${def.id}');
  }

  /// 标记“模型调用位置的文件”为自选外部模型（写 prefs + 刷新缓存路径）。
  static Future<void> markCustomModelActive(String displayName) async {
    final def = selectedDef;
    final dir = await getApplicationSupportDirectory();
    final targetPath = path.join(dir.path, def.fileName);
    _cachedModelPath = targetPath;
    _customModelActive = true;
    _customModelName = displayName;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('anime4kV4_custom_${def.id}', true);
    await prefs.setString('anime4kV4_custom_name_${def.id}', displayName);
  }

  /// 把打包进 APK 的模型（assets）抽取到应用目录（模型调用位置），仅在需要时执行一次。
  ///
  /// 触发条件（全部满足才回灌）：
  ///  1. 未启用“自选外部模型”（用户自选优先，绝不覆盖）；
  ///  2. 模型调用位置当前没有有效文件（不存在或体积 < 下限）；
  ///  3. 尚未标记为“打包模型已安装”——用户手动删除后不自动回灌，尊重用户意愿。
  ///
  /// assets 中缺少该模型（例如 2x 模型未打包）时静默跳过，让下载管理器接管，保证 APK 始终可构建。
  static Future<void> extractBundledModelIfNeeded() async {
    try {
      final def = selectedDef;
      // 用户自选外部模型时不覆盖
      if (await isCustomModelActive()) return;

      final prefs = await SharedPreferences.getInstance();
      final installedKey = 'anime4kV4_bundled_${def.id}';
      if (prefs.getBool(installedKey) ?? false) {
        // 已抽取过一次；若用户随后手动删除，不再自动回灌
        return;
      }

      final dir = await getApplicationSupportDirectory();
      final targetPath = path.join(dir.path, def.fileName);
      final targetFile = File(targetPath);

      // 已有有效模型（下载/导入）则无需抽取，直接标记完成
      if (await targetFile.exists() &&
          await targetFile.length() > _validModelMinSize) {
        await prefs.setBool(installedKey, true);
        return;
      }

      // 未打包（2x 模型默认不打包）则静默跳过，交给下载管理器
      final assetPath = def.bundledAssetPath;
      if (assetPath == null) return;

      // 从 assets 读取打包模型；不存在（未打包）则静默跳过
      final ByteData data;
      try {
        data = await rootBundle.load(assetPath);
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

      await prefs.setBool(installedKey, true);
      _cachedModelPath = targetPath;
      Log.info('Anime4KV4',
          'bundled model (${def.id}) extracted to $targetPath (${bytes.length} bytes)');
    } catch (e, s) {
      Log.error('Anime4KV4', 'extractBundledModelIfNeeded failed: $e\n$s');
    }
  }

  /// 获取模型文件路径（即模型调用位置）。不自动下载；文件不存在或无效则返回 null。
  static Future<String?> ensureModelAvailable() async {
    final def = selectedDef;
    if (_cachedModelPath != null) {
      final f = File(_cachedModelPath!);
      if (await f.exists() && await f.length() > _validModelMinSize) {
        return _cachedModelPath!;
      }
      _cachedModelPath = null;
    }

    final dir = await getApplicationSupportDirectory();
    final targetPath = path.join(dir.path, def.fileName);
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
    final def = selectedDef;
    final dir = await getApplicationSupportDirectory();
    final targetPath = path.join(dir.path, def.fileName);
    final targetFile = File(targetPath);
    if (await targetFile.exists()) {
      final size = await targetFile.length();
      if (size > _validModelMinSize) return true;
    }
    if (_cachedModelPath != null) return true;
    return false;
  }

  static Future<int> getDownloadedSize() async {
    final def = selectedDef;
    final dir = await getApplicationSupportDirectory();
    final targetPath = path.join(dir.path, def.fileName);
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

    final def = selectedDef;
    final dir = await getApplicationSupportDirectory();
    final targetPath = path.join(dir.path, def.fileName);
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
            await prefs.setBool('anime4kV4_custom_${def.id}', false);
            await prefs.remove('anime4kV4_custom_name_${def.id}');
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
    final def = selectedDef;
    final dir = await getApplicationSupportDirectory();
    final targetPath = path.join(dir.path, def.fileName);
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
    await prefs.setBool('anime4kV4_custom_${def.id}', false);
    await prefs.remove('anime4kV4_custom_name_${def.id}');
  }

  /// 通过原生 ContentResolver（colorize 通道的 copyUri）把 content URI / 文件路径以
  /// 有界分块（64KB）方式拷贝到模型调用位置，避免一次性读入内存（OOM）或拷坏。
  /// 实际文件拷贝由 [ColorizationService.copyUriTo] 经原生完成。
  static Future<int> copyUriTo(String uri, String dest) =>
      ColorizationService.instance.copyUriTo(uri, dest);
}
