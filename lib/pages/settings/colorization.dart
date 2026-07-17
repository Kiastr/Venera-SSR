part of 'settings_page.dart';

/// 图像上色设置页
///
/// 参考 Anime4K 设置页的模式。提供开关、强度调节和模型管理。
class ColorizationSettings extends StatefulWidget {
  const ColorizationSettings({super.key});

  @override
  State<ColorizationSettings> createState() => _ColorizationSettingsState();
}

class _ColorizationSettingsState extends State<ColorizationSettings> {
  bool _isModelDownloaded = false;
  bool _isDownloading = false;
  double _downloadProgress = 0.0;
  String _status = '';
  String? _customModelName;
  List<String> _modelUrls = [];
  bool _usingCustom = false;
  String _selectedVariant = 'deoldify';

  @override
  void initState() {
    super.initState();
    _refreshModelStatus();
  }

  Future<void> _refreshModelStatus() async {
    final usingCustom = await ColorizationModelManager.isCustomModelActive();
    final customName = await ColorizationModelManager.getCustomModelName();
    final urls = await ColorizationModelManager.getModelUrls();
    final downloaded = await ColorizationModelManager.isModelDownloaded;
    final variant = await ColorizationModelManager.getSelectedVariant();
    if (mounted) {
      setState(() {
        _customModelName = customName;
        _modelUrls = urls;
        _isModelDownloaded = downloaded;
        _usingCustom = usingCustom;
        _selectedVariant = variant;
      });
    }
  }

  String get _selectedVariantLabel {
    final v = ColorizationModelManager.modelVariants.firstWhere(
      (e) => e.id == _selectedVariant,
      orElse: () => const ColorizationModelVariant('deoldify', 'DeOldify Artistic'),
    );
    return v.label;
  }

  Future<void> _downloadModel() async {
    if (_isDownloading) return;
    setState(() {
      _isDownloading = true;
      _downloadProgress = 0.0;
      _status = 'Preparing...';
    });

    try {
      await ColorizationModelManager.downloadModel(
        variant: _selectedVariant,
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
        context.showMessage(message: "Download failed: @e".tlParams({'e': e.toString()}));
      }
    } finally {
      _isDownloading = false;
      // 模型文件已变更，失效原生会话缓存并刷新服务路径缓存
      await ColorizationService.instance.resetNativeSession();
      await ColorizationService.instance.checkModelAvailable();
      await _refreshModelStatus();
      if (mounted) {
        setState(() {
          if (_isModelDownloaded) {
            _status = 'Ready';
          } else {
            _status = '';
          }
        });
      }
    }
  }

  Future<void> _deleteModel() async {
    await ColorizationModelManager.clearModel();
    await ColorizationService.instance.clearCache();
    // 让服务感知模型已删除（重置 _modelPath，校验文件不存在）
    await ColorizationService.instance.resetNativeSession();
    await ColorizationService.instance.checkModelAvailable();
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
      // 直接落到模型调用位置 deoldify_artistic.onnx。
      // 这是“选择外部模型崩溃”的根治：openRead() 在 content URI 下会把整文件读入内存
      // （OOM）或产出损坏文件（原生 createSession 读到坏模型 → segfault）。
      final uri = xFile.path; // content URI 或真实文件路径
      final dir = await getApplicationSupportDirectory();
      final targetPath = path.join(dir.path, ColorizationModelManager.modelFileName);
      final bakPath = '$targetPath.bak';
      final tempPath = '$targetPath.tmp';

      // 已存在下载模型则先备份，便于“回退内置模型”还原
      final targetFile = File(targetPath);
      if (await targetFile.exists()) {
        await targetFile.rename(bakPath);
      }

      int written;
      try {
        written = await ColorizationService.instance.copyUriTo(uri, tempPath);
      } catch (e) {
        // 拷贝失败：还原备份
        if (await File(bakPath).exists()) await File(bakPath).rename(targetPath);
        if (mounted) context.showMessage(message: "Failed to copy file: @e".tlParams({'e': e.toString()}));
        return;
      }

      if (written < ColorizationModelManager.validModelMinSize) {
        await File(tempPath).delete().catchError((_) {});
        if (await File(bakPath).exists()) await File(bakPath).rename(targetPath);
        if (mounted) context.showMessage(message: "File too small, invalid model".tl);
        return;
      }

      await File(tempPath).rename(targetPath);
      await File(bakPath).delete().catchError((_) {});

      // 记账为自选模型 + 失效原生会话缓存 + 让服务立即感知新路径
      await ColorizationModelManager.markCustomModelActive(xFile.name);
      await ColorizationService.instance.resetNativeSession();
      await ColorizationService.instance.checkModelAvailable();
      await _refreshModelStatus();
      if (mounted) context.showMessage(message: "Custom model selected".tl);
    } catch (e) {
      if (mounted) context.showMessage(message: "Failed to pick file: @e".tlParams({'e': e.toString()}));
    }
  }

  /// 清除自选模型，回退到内置（下载）模型
  Future<void> _clearCustomModel() async {
    await ColorizationModelManager.clearCustomModelSelection();
    await ColorizationService.instance.resetNativeSession();
    await ColorizationService.instance.checkModelAvailable();
    await _refreshModelStatus();
    if (mounted) context.showMessage(message: "Reverted to built-in model".tl);
  }

  /// 添加一个自定义镜像 URL
  Future<void> _addMirrorUrl() async {
    await showInputDialog(
      context: context,
      title: "Add Mirror URL".tl,
      hintText: "https://.../deoldify.onnx",
      confirmText: "Add".tl,
      onConfirm: (url) async {
        await ColorizationModelManager.addModelUrl(url);
        await _refreshModelStatus();
        return null as Object?;
      },
    );
  }

  /// 删除指定下标的镜像 URL
  Future<void> _removeMirrorUrl(int index) async {
    await ColorizationModelManager.removeModelUrlAt(index);
    await _refreshModelStatus();
  }

  @override
  Widget build(BuildContext context) {
    return SmoothCustomScrollView(
      slivers: [
        SliverAppbar(title: Text("Colorization".tl)),
        _SwitchSetting(
          title: "Enable Image Colorization".tl,
          subtitle:
              _isModelDownloaded
                  ? "Model is ready".tl
                  : "Download model below to enable".tl,
          settingKey: "enableColorization",
        ).toSliver(),
        _SliderSetting(
          title: "Colorization Intensity".tl,
          settingsIndex: "colorizationIntensity",
          min: 0.3,
          max: 1.2,
          interval: 0.05,
        ).toSliver(),
        // 模型管理区域
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
                  Row(
                    children: [
                      Expanded(
                        child: Text(
                          _selectedVariantLabel,
                          style: const TextStyle(fontWeight: FontWeight.bold),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  // 模型变体选择器（仅切换下载源，推理逻辑不变）
                  Wrap(
                    spacing: 8,
                    children:
                        ColorizationModelManager.modelVariants.map((v) {
                          final selected = _selectedVariant == v.id;
                          return ChoiceChip(
                            label: Text(v.label.tl),
                            selected: selected,
                            onSelected: (_) async {
                              await ColorizationModelManager.setSelectedVariant(
                                v.id,
                              );
                              await _refreshModelStatus();
                            },
                          );
                        }).toList(),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    _isModelDownloaded
                        ? "Model downloaded".tl
                        : (_selectedVariant == 'deoldify-int8'
                            ? "Model not downloaded (lightweight)".tl
                            : "Model not downloaded (~243MB)".tl),
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
                            icon:
                                _isDownloading
                                    ? const SizedBox(
                                      width: 16,
                                      height: 16,
                                      child: CircularProgressIndicator(
                                        strokeWidth: 2,
                                      ),
                                    )
                                    : const Icon(Icons.download),
                            label: Text(
                              _isDownloading ? "Downloading...".tl : "Download Model".tl,
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
                  Row(
                    children: [
                      Text(
                        "Custom Model File".tl,
                        style: const TextStyle(fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(width: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                        decoration: BoxDecoration(
                          color: Colors.orange.withValues(alpha: 0.15),
                          borderRadius: BorderRadius.circular(4),
                          border: Border.all(
                            color: Colors.orange.withValues(alpha: 0.4),
                          ),
                        ),
                        child: Text(
                          "实验性".tl,
                          style: TextStyle(
                            fontSize: 10,
                            color: Colors.orange.shade700,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ],
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
                      await ColorizationModelManager.resetModelUrls();
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
        ListTile(
          title: Text("Clear Colorization Cache".tl),
          trailing: const Icon(Icons.delete_sweep),
          onTap: () async {
            await ColorizationService.instance.clearCache();
            if (mounted) {
              context.showMessage(message: "Colorization cache cleared".tl);
            }
          },
        ).toSliver(),
      ],
    );
  }
}

/// 镜像 URL 列表项
class _MirrorUrlTile extends StatelessWidget {
  final int index;
  final String url;
  final void Function(int) onDelete;

  const _MirrorUrlTile({
    required this.index,
    required this.url,
    required this.onDelete,
  });

  @override
  Widget build(BuildContext context) {
    return SliverToBoxAdapter(
      child: ListTile(
        dense: true,
        leading: Text('${index + 1}'),
        title: Text(
          url,
          style: const TextStyle(fontSize: 12),
          maxLines: 2,
          overflow: TextOverflow.ellipsis,
        ),
        trailing: IconButton(
          icon: const Icon(Icons.delete_outline, size: 20),
          onPressed: () => onDelete(index),
        ),
      ),
    );
  }
}
