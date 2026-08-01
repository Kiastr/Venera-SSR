part of 'settings_page.dart';

/// Anime4K 设置页
///
/// 同时管理两个引擎版本：
///  - v1：纯 Dart CPU 算法（Gauss/Unblur/GradientRefine），缩放 1–4x，无模型文件；
///  - v4：Anime4K v4 超分 ONNX 模型（默认官方 ACNet 2×，可选 Real-ESRGAN 4× / 通用 2×），经原生
///    ONNX Runtime + NNAPI(GPU) 超分，倍数由模型实际维度决定（getScale 探测），仅 Android 生效。
///
/// 两版本并存，由 `anime4KVersion` 设置选择；v4 选中时显示模型管理卡片，并隐藏 v1 专用滑块。
class Anime4KSettings extends StatefulWidget {
  const Anime4KSettings({super.key});

  @override
  State<Anime4KSettings> createState() => _Anime4KSettingsState();
}

class _Anime4KSettingsState extends State<Anime4KSettings> {
  bool _isModelDownloaded = false;
  bool _isDownloading = false;
  double _downloadProgress = 0.0;
  String _status = '';
  String? _customModelName;
  List<String> _modelUrls = [];
  bool _usingCustom = false;

  @override
  void initState() {
    super.initState();
    _refreshModelStatus();
  }

  Future<void> _refreshModelStatus() async {
    final usingCustom = await Anime4KV4ModelManager.isCustomModelActive();
    final customName = await Anime4KV4ModelManager.getCustomModelName();
    final urls = await Anime4KV4ModelManager.getModelUrls();
    final downloaded = await Anime4KV4ModelManager.isModelDownloaded;
    if (mounted) {
      setState(() {
        _customModelName = customName;
        _modelUrls = urls;
        _isModelDownloaded = downloaded;
        _usingCustom = usingCustom;
      });
    }
  }

  String get _version => appdata.settings['anime4KVersion'] as String? ?? 'v1';

  void _setVersion(String v) {
    if (_version == v) return;
    appdata.settings['anime4KVersion'] = v;
    appdata.saveData();
    // 切换引擎后强制刷新图片（v1/v4 输出不同，缓存不可复用）
    PaintingBinding.instance.imageCache.clear();
    ComicImage.clear();
    setState(() {});
  }

  /// 切换 v4 超分模型（4x/2x）。不同倍数输出尺寸不同，清缓存避免串图。
  Future<void> _selectModel(String id) async {
    if (Anime4KV4ModelManager.selectedDef.id == id) return;
    await Anime4KV4Service.instance.setModel(id);
    PaintingBinding.instance.imageCache.clear();
    ComicImage.clear();
    await _refreshModelStatus();
    if (mounted) setState(() {});
  }

  Future<void> _downloadModel() async {
    if (_isDownloading) return;
    setState(() {
      _isDownloading = true;
      _downloadProgress = 0.0;
      _status = 'Preparing...';
    });

    try {
      await Anime4KV4ModelManager.downloadModel(
        onProgress: (progress) {
          if (mounted) {
            setState(() {
              _downloadProgress = progress;
              _status = 'Downloading ${(progress * 100).toStringAsFixed(1)}%';
            });
          }
        },
        onStatus: (status) {
          if (mounted) {
            setState(() {
              _status = status;
            });
          }
        },
      );
      if (mounted) {
        context.showMessage(message: "Model downloaded".tl);
      }
    } catch (e) {
      if (mounted) {
        context.showMessage(message: "Download failed: $e".tl);
      }
    } finally {
      _isDownloading = false;
      // 模型文件已变更，失效原生会话缓存并刷新服务路径缓存
      await Anime4KV4Service.instance.resetNativeSession();
      await Anime4KV4Service.instance.checkModelAvailable();
      await _refreshModelStatus();
      if (mounted) {
        setState(() {
          _status = _isModelDownloaded ? 'Ready' : '';
        });
      }
    }
  }

  Future<void> _deleteModel() async {
    await Anime4KV4ModelManager.clearModel();
    await Anime4KV4Service.instance.clearCache();
    // 让服务感知模型已删除（重置 _modelPath，校验文件不存在）
    await Anime4KV4Service.instance.resetNativeSession();
    await Anime4KV4Service.instance.checkModelAvailable();
    await _refreshModelStatus();
    if (mounted) {
      context.showMessage(message: "Model deleted".tl);
      setState(() {
        _status = '';
        _downloadProgress = 0.0;
      });
    }
  }

  /// 选择本地 .onnx 模型文件（优先级高于内置下载模型）
  Future<void> _pickLocalModel() async {
    try {
      final xFile = await file_selector.openFile(
        acceptedTypeGroups: <file_selector.XTypeGroup>[
          file_selector.XTypeGroup(
            label: 'ONNX Model',
            extensions: ['onnx'],
          ),
        ],
      );
      if (xFile == null) return;
      if (!xFile.name.toLowerCase().endsWith('.onnx')) {
        if (mounted) context.showMessage(message: "Please select a .onnx file".tl);
        return;
      }
      // 通过原生 ContentResolver 以 64KB 分块拷贝（不占内存、不拷坏），
      // 直接落到当前选中模型的调用位置（fileName）。
      final uri = xFile.path; // content URI 或真实文件路径
      final dir = await getApplicationSupportDirectory();
      final targetPath = path.join(dir.path, Anime4KV4ModelManager.modelFileName);
      final bakPath = '$targetPath.bak';
      final tempPath = '$targetPath.tmp';

      // 已存在下载模型则先备份，便于“回退内置模型”还原
      final targetFile = File(targetPath);
      if (await targetFile.exists()) {
        await targetFile.rename(bakPath);
      }

      int written;
      try {
        written = await Anime4KV4ModelManager.copyUriTo(uri, tempPath);
      } catch (e) {
        if (await File(bakPath).exists()) await File(bakPath).rename(targetPath);
        if (mounted) context.showMessage(message: "Failed to copy file: $e".tl);
        return;
      }

      if (written < Anime4KV4ModelManager.validModelMinSize) {
        await File(tempPath).delete().catchError((_) {});
        if (await File(bakPath).exists()) await File(bakPath).rename(targetPath);
        if (mounted) context.showMessage(message: "File too small, invalid model".tl);
        return;
      }

      await File(tempPath).rename(targetPath);
      await File(bakPath).delete().catchError((_) {});

      // 记账为自选模型 + 失效原生会话缓存 + 让服务立即感知新路径
      await Anime4KV4ModelManager.markCustomModelActive(xFile.name);
      await Anime4KV4Service.instance.resetNativeSession();
      await Anime4KV4Service.instance.checkModelAvailable();
      await _refreshModelStatus();
      if (mounted) context.showMessage(message: "Custom model selected".tl);
    } catch (e) {
      if (mounted) context.showMessage(message: "Failed to pick file: $e".tl);
    }
  }

  /// 清除自选模型，回退到内置（下载）模型
  Future<void> _clearCustomModel() async {
    await Anime4KV4ModelManager.clearCustomModelSelection();
    await Anime4KV4Service.instance.resetNativeSession();
    await Anime4KV4Service.instance.checkModelAvailable();
    await _refreshModelStatus();
    if (mounted) context.showMessage(message: "Reverted to built-in model".tl);
  }

  /// 添加一个自定义镜像 URL
  Future<void> _addMirrorUrl() async {
    await showInputDialog(
      context: context,
      title: "Add Mirror URL".tl,
      hintText: "https://.../${Anime4KV4ModelManager.modelFileName}",
      confirmText: "Add".tl,
      onConfirm: (url) async {
        await Anime4KV4ModelManager.addModelUrl(url);
        await _refreshModelStatus();
        return null as Object?;
      },
    );
  }

  /// 删除指定下标的镜像 URL
  Future<void> _removeMirrorUrl(int index) async {
    await Anime4KV4ModelManager.removeModelUrlAt(index);
    await _refreshModelStatus();
  }

  @override
  Widget build(BuildContext context) {
    final isV4 = _version == 'v4';
    return SmoothCustomScrollView(
      slivers: [
        SliverAppbar(title: Text("Anime4K".tl)),
        _SwitchSetting(
          title: "Enable Anime4K Upscaling".tl,
          settingKey: "enableAnime4K",
          beforeChange: (newValue) async {
            // 关闭或 v1 直接放行
            if (!newValue) return true;
            if (_version != 'v4') return true;
            // v4 开启前必须确保模型已下载
            final downloaded = await Anime4KV4ModelManager.isModelDownloaded;
            if (downloaded) return true;
            if (!mounted) return false;
            final confirm = await showDialog<bool>(
              context: context,
              builder: (dialogContext) {
                return ContentDialog(
                  title: "Model Required".tl,
                  content: Text(
                    "Anime4K v4 model (${Anime4KV4ModelManager.selectedDef.displayName}) is not downloaded. Download (~${Anime4KV4ModelManager.selectedDef.sizeHintMB}MB) to enable?"
                        .tl,
                  ).paddingHorizontal(16).fixWidth(double.infinity),
                  actions: [
                    Button.filled(
                      onPressed: () => dialogContext.pop(true),
                      child: Text("Download".tl),
                    ),
                    Button.outlined(
                      onPressed: () => dialogContext.pop(false),
                      child: Text("Cancel".tl),
                    ),
                  ],
                );
              },
            );
            if (confirm == true) {
              await _downloadModel();
              if (mounted && await Anime4KV4ModelManager.isModelDownloaded) {
                appdata.settings['enableAnime4K'] = true;
                appdata.saveData();
                PaintingBinding.instance.imageCache.clear();
                ComicImage.clear();
                setState(() {});
              }
            }
            // 由上面的手动置位控制开关，拦截这次手势
            return false;
          },
        ).toSliver(),
        // 引擎版本选择
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
            child: Text(
              "Engine Version".tl,
              style: TextStyle(
                color: context.colorScheme.primary,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ),
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
            child: Wrap(
              spacing: 8,
              children: [
                ChoiceChip(
                  label: Text("v1 (CPU)".tl),
                  selected: !isV4,
                  onSelected: (_) => _setVersion('v1'),
                ),
                ChoiceChip(
                  label: Text("v4 (AI · GPU)".tl),
                  selected: isV4,
                  onSelected: (_) => _setVersion('v4'),
                ),
              ],
            ),
          ),
        ),
        // v4 模型（倍数）选择：4x 动画 / 2x 通用
        if (isV4)
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
              child: Wrap(
                spacing: 8,
                children: Anime4KV4ModelManager.getModels().map((m) {
                  final selected = Anime4KV4ModelManager.selectedDef.id == m.id;
                  return ChoiceChip(
                    label: Text("${m.scale}×  ${m.displayName}".tl),
                    selected: selected,
                    onSelected: (_) => _selectModel(m.id),
                  );
                }).toList(),
              ),
            ),
          ),
        // v1 专用参数（Scale/Push/Grad）：仅 v1 显示
        SliverAnimatedVisibility(
          visible: !isV4,
          child: Column(
            children: [
              _SliderSetting(
                title: "Scale Factor".tl,
                settingsIndex: "anime4KScaleFactor",
                min: 1.0,
                max: 4.0,
                interval: 0.5,
              ),
              _SliderSetting(
                title: "Push Strength".tl,
                settingsIndex: "anime4KPushStrength",
                min: 0.0,
                max: 1.0,
                interval: 0.05,
              ),
              _SliderSetting(
                title: "Gradient Refine Strength".tl,
                settingsIndex: "anime4KPushGradStrength",
                min: 0.0,
                max: 1.0,
                interval: 0.05,
              ),
            ],
          ),
        ),
        // v4 强度（倍数由模型决定，仅调节强度）
        SliverAnimatedVisibility(
          visible: isV4,
          child: _SliderSetting(
            title:
                "Upscale Intensity (v4 · ${Anime4KV4ModelManager.selectedDef.scale}x)"
                    .tl,
            settingsIndex: "anime4KV4Intensity",
            min: 0.3,
            max: 1.2,
            interval: 0.05,
          ),
        ),
        ListTile(
          title: Text("Clear Anime4K Cache".tl),
          trailing: const Icon(Icons.delete_sweep),
          onTap: () async {
            await Anime4KService.instance.clearCache();
            if (isV4) await Anime4KV4Service.instance.clearCache();
            if (mounted) {
              context.showMessage(message: "Anime4K cache cleared".tl);
            }
          },
        ).toSliver(),
        // ---- v4 模型管理（仅 v4 显示） ----
        if (isV4) ...[
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
              child: Text(
                "Model Management".tl,
                style: TextStyle(
                  color: context.colorScheme.primary,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),
          SliverToBoxAdapter(
            child: Card(
              margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      Anime4KV4ModelManager.selectedDef.displayName.tl,
                      style: const TextStyle(fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      _isModelDownloaded
                          ? "Model downloaded".tl
                          : "Model not downloaded (~${Anime4KV4ModelManager.selectedDef.sizeHintMB}MB)"
                              .tl,
                      style: TextStyle(
                        color: context.colorScheme.onSurfaceVariant,
                        fontSize: 12,
                      ),
                    ),
                    if (_status.isNotEmpty) ...[
                      const SizedBox(height: 8),
                      Text(
                        _status,
                        style: TextStyle(
                          color: context.colorScheme.primary,
                          fontSize: 12,
                        ),
                      ),
                    ],
                    if (_isDownloading) ...[
                      const SizedBox(height: 8),
                      LinearProgressIndicator(value: _downloadProgress),
                    ],
                    const SizedBox(height: 12),
                    if (!_usingCustom)
                      Row(
                        children: [
                          if (!_isModelDownloaded)
                            Expanded(
                              child: ElevatedButton.icon(
                                onPressed:
                                    _isDownloading ? null : _downloadModel,
                                icon: _isDownloading
                                    ? const SizedBox(
                                        width: 16,
                                        height: 16,
                                        child: CircularProgressIndicator(
                                          strokeWidth: 2,
                                        ),
                                      )
                                    : const Icon(Icons.download),
                                label: Text(
                                  _isDownloading
                                      ? "Downloading...".tl
                                      : "Download Model".tl,
                                ),
                              ),
                            ),
                          if (_isModelDownloaded) ...[
                            Expanded(
                              child: OutlinedButton.icon(
                                onPressed: _deleteModel,
                                icon: const Icon(Icons.delete_outline),
                                label: Text("Delete Model".tl),
                              ),
                            ),
                          ],
                        ],
                      ),
                  ],
                ),
              ),
            ),
          ),
          // 自选本地模型文件
          SliverToBoxAdapter(
            child: Card(
              margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      "Custom Model File".tl,
                      style: const TextStyle(fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      _usingCustom
                          ? "Using: ${_customModelName ?? 'custom model'}".tl
                          : "Select a local .onnx model to override the built-in one"
                              .tl,
                      style: TextStyle(
                        color: context.colorScheme.onSurfaceVariant,
                        fontSize: 12,
                      ),
                    ),
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        Expanded(
                          child: ElevatedButton.icon(
                            onPressed: _pickLocalModel,
                            icon: const Icon(Icons.folder_open),
                            label: Text("Select Model File".tl),
                          ),
                        ),
                        if (_usingCustom) ...[
                          const SizedBox(width: 8),
                          Expanded(
                            child: OutlinedButton.icon(
                              onPressed: _clearCustomModel,
                              icon: const Icon(Icons.restore),
                              label: Text("Use Built-in".tl),
                            ),
                          ),
                        ],
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
          // 镜像 URL 管理
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
              child: Text(
                "Download Mirrors".tl,
                style: TextStyle(
                  color: context.colorScheme.primary,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),
          ..._modelUrls.asMap().entries.map(
                (e) => _MirrorUrlTile(
                  index: e.key,
                  url: e.value,
                  onDelete: _removeMirrorUrl,
                ),
              ),
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: _addMirrorUrl,
                      icon: const Icon(Icons.add),
                      label: Text("Add Mirror URL".tl),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: () async {
                        await Anime4KV4ModelManager.resetModelUrls();
                        await _refreshModelStatus();
                      },
                      icon: const Icon(Icons.restart_alt),
                      label: Text("Reset".tl),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ],
    );
  }
}

