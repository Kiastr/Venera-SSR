part of 'reader.dart';

class ComicImage extends StatefulWidget {
  /// Modified from flutter Image
  ComicImage({
    required ImageProvider image,
    super.key,
    double scale = 1.0,
    this.semanticLabel,
    this.excludeFromSemantics = false,
    this.width,
    this.height,
    this.color,
    this.opacity,
    this.colorBlendMode,
    this.fit,
    this.alignment = Alignment.center,
    this.repeat = ImageRepeat.noRepeat,
    this.centerSlice,
    this.matchTextDirection = false,
    this.gaplessPlayback = false,
    this.filterQuality = FilterQuality.medium,
    this.isAntiAlias = false,
    Map<String, String>? headers,
    int? cacheWidth,
    int? cacheHeight,
    this.onInit,
    this.onDispose,
  })  : image = ResizeImage.resizeIfNeeded(cacheWidth, cacheHeight, image),
        assert(cacheWidth == null || cacheWidth > 0),
        assert(cacheHeight == null || cacheHeight > 0);

  final ImageProvider image;

  final String? semanticLabel;

  final bool excludeFromSemantics;

  final double? width;

  final double? height;

  final bool gaplessPlayback;

  final bool matchTextDirection;

  final Rect? centerSlice;

  final ImageRepeat repeat;

  final AlignmentGeometry alignment;

  final BoxFit? fit;

  final BlendMode? colorBlendMode;

  final FilterQuality filterQuality;

  final Animation<double>? opacity;

  final Color? color;

  final bool isAntiAlias;

  final void Function(State<ComicImage> state)? onInit;

  final void Function(State<ComicImage> state)? onDispose;

  static void clear() => _ComicImageState.clear();

  @override
  State<ComicImage> createState() => _ComicImageState();
}

class _ComicImageState extends State<ComicImage> with WidgetsBindingObserver {
  ImageStream? _imageStream;
  ImageInfo? _imageInfo;
  ImageChunkEvent? _loadingProgress;
  bool _isListeningToStream = false;
  late bool _invertColors;
  int? _frameNumber;
  bool _wasSynchronouslyLoaded = false;
  late DisposableBuildContext<State<ComicImage>> _scrollAwareContext;
  Object? _lastException;
  ImageStreamCompleterHandle? _completerHandle;
  
  // Anime4K 超分相关变量
  Uint8List? _upscaledBytes;
  bool _isUpscaling = false;

  // 图像上色相关变量
  Uint8List? _colorizedBytes;
  bool _isColorizing = false;

  static final Map<int, Size> _cache = {};

  /// 追踪所有活跃的实例，用于在 clear() 时重置所有实例的处理状态
  static final Set<_ComicImageState> _instances = {};

  static void clear() {
    _cache.clear();
    // 重置所有活跃实例的处理状态，并重新触发处理。
    // 否则 didUpdateWidget 会因 widget.image 未变而不再触发，
    // 导致开关切换后已显示的图片不重新处理（超分/上色不生效）。
    for (final instance in _instances) {
      if (instance.mounted) {
        instance._upscaledBytes = null;
        instance._isUpscaling = false;
        instance._colorizedBytes = null;
        instance._isColorizing = false;
        instance._triggerImageUpscale();
        instance._triggerImageColorization();
      }
    }
  }

  @override
  void initState() {
    super.initState();
    _instances.add(this);
    WidgetsBinding.instance.addObserver(this);
    _scrollAwareContext = DisposableBuildContext<State<ComicImage>>(this);
    widget.onInit?.call(this);
  }

  @override
  void dispose() {
    assert(_imageStream != null);
    _instances.remove(this);
    WidgetsBinding.instance.removeObserver(this);
    _stopListeningToStream();
    _completerHandle?.dispose();
    _scrollAwareContext.dispose();
    _replaceImage(info: null);
    widget.onDispose?.call(this);
    super.dispose();
  }

  @override
  void didChangeDependencies() {
    _updateInvertColors();
    _resolveImage();
    _triggerImageUpscale();
    _triggerImageColorization();

    if (TickerMode.of(context)) {
      _listenToStream();
    } else {
      _stopListeningToStream(keepStreamAlive: true);
    }

    super.didChangeDependencies();
  }

  @override
  void didUpdateWidget(ComicImage oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.image != oldWidget.image) {
      // 当图像更换时，必须重置已处理的字节数据和处理状态，
      // 否则 _upscaledBytes/_colorizedBytes 仍指向旧图的结果，
      // 触发方法会因 "!= null" 提前返回，导致新图永远不会被处理。
      _upscaledBytes = null;
      _colorizedBytes = null;
      _isUpscaling = false;
      _isColorizing = false;
      _resolveImage();
      _triggerImageUpscale();
      _triggerImageColorization();
    }
  }

  @override
  void didChangeAccessibilityFeatures() {
    super.didChangeAccessibilityFeatures();
    setState(() {
      _updateInvertColors();
    });
  }

  @override
  void reassemble() {
    _resolveImage(); // in case the image cache was flushed
    super.reassemble();
  }

  bool containsPoint(Offset point) {
    if (!mounted) {
      return false;
    }
    var renderBox = context.findRenderObject() as RenderBox;
    var localPoint = renderBox.globalToLocal(point);
    return renderBox.paintBounds.contains(localPoint);
  }

  void _updateInvertColors() {
    _invertColors = MediaQuery.maybeInvertColorsOf(context) ??
        SemanticsBinding.instance.accessibilityFeatures.invertColors;
  }

  /// 触发图像超分处理
  Future<void> _triggerImageUpscale() async {
    final provider = _getReaderImageProvider();

    // 检查是否启用了 Anime4K
    bool enableAnime4K;
    if (provider != null) {
      enableAnime4K = appdata.settings.getReaderSetting(
        provider.cid,
        provider.sourceKey ?? "",
        'enableAnime4K',
      ) == true;
    } else {
      enableAnime4K = appdata.settings['enableAnime4K'] == true;
    }

    if (!enableAnime4K) {
      if (_upscaledBytes != null) {
        setState(() {
          _upscaledBytes = null;
        });
      }
      return;
    }

    if (_upscaledBytes != null || _isUpscaling) return;

    // 如果没有可用的图像 provider，不标记为正在处理以避免永久卡住
    if (provider == null) return;

    _isUpscaling = true;

    try {
      final chunkController = StreamController<ImageChunkEvent>();
      chunkController.stream.listen(null, onError: (_) {});
      final imageBytes = await provider.load(
        chunkController,
        () {},
      );
      unawaited(chunkController.close());

      if (imageBytes.isEmpty) {
        Log.warning('ComicImage', 'Anime4K: empty image bytes for ${provider.key}');
        _isUpscaling = false;
        return;
      }

      // 加载过程中 widget 可能已被销毁
      if (!mounted) {
        _isUpscaling = false;
        return;
      }

      Log.info('ComicImage', 'Anime4K: start processing ${provider.key}');

      final result = await Anime4KService.instance.processImage(
        imageBytes: imageBytes,
        cacheKey: provider.key,
        scaleFactor: (appdata.settings.getReaderSetting(provider.cid, provider.sourceKey ?? "", 'anime4KScaleFactor') as num?)?.toDouble() ?? 2.0,
        pushStrength: (appdata.settings.getReaderSetting(provider.cid, provider.sourceKey ?? "", 'anime4KPushStrength') as num?)?.toDouble() ?? 0.31,
        pushGradStrength: (appdata.settings.getReaderSetting(provider.cid, provider.sourceKey ?? "", 'anime4KPushGradStrength') as num?)?.toDouble() ?? 1.0,
      );

      if (!mounted) {
        _isUpscaling = false;
        return;
      }

      if (result != null) {
        Log.info('ComicImage', 'Anime4K: done ${provider.key}');
        setState(() {
          _upscaledBytes = result;
          _isUpscaling = false;
        });
      } else {
        Log.warning('ComicImage', 'Anime4K: null result for ${provider.key}');
        _isUpscaling = false;
      }
    } catch (e, s) {
      Log.error('ComicImage', 'Anime4K processing error: $e', s);
      _isUpscaling = false;
    }
  }

  /// 触发图像上色处理
  Future<void> _triggerImageColorization() async {
    final provider = _getReaderImageProvider();

    // 检查是否启用了上色功能
    bool enableColorization;
    if (provider != null) {
      enableColorization = appdata.settings.getReaderSetting(
            provider.cid,
            provider.sourceKey ?? "",
            'enableColorization',
          ) ==
          true;
    } else {
      enableColorization = appdata.settings['enableColorization'] == true;
    }

    if (!enableColorization) {
      if (_colorizedBytes != null) {
        setState(() {
          _colorizedBytes = null;
        });
      }
      return;
    }

    if (_colorizedBytes != null || _isColorizing) return;

    // 如果没有可用的图像 provider，不标记为正在处理以避免永久卡住
    if (provider == null) return;

    // 模型未就绪时重新检测一次（用户可能刚下载模型但服务尚未刷新），
    // 仍不可用则静默跳过（避免卡住 _isColorizing）
    if (!ColorizationService.instance.isModelAvailable) {
      if (!await ColorizationService.instance.checkModelAvailable()) {
        return;
      }
    }

    _isColorizing = true;

    try {
      final chunkController = StreamController<ImageChunkEvent>();
      chunkController.stream.listen(null, onError: (_) {});
      final imageBytes = await provider.load(
        chunkController,
        () {},
      );
      unawaited(chunkController.close());

      if (imageBytes.isEmpty) {
        Log.warning('ComicImage', 'Colorization: empty image bytes for ${provider.key}');
        _isColorizing = false;
        return;
      }

      // 加载过程中 widget 可能已被销毁
      if (!mounted) {
        _isColorizing = false;
        return;
      }

      Log.info('ComicImage', 'Colorization: start processing ${provider.key}');

      final result = await ColorizationService.instance.processImage(
        imageBytes: imageBytes,
        cacheKey: provider.key,
        intensity: (appdata.settings.getReaderSetting(
              provider.cid,
              provider.sourceKey ?? "",
              'colorizationIntensity',
            ) as num?)?.toDouble() ?? 1.0,
      );

      if (!mounted) {
        _isColorizing = false;
        return;
      }

      if (result != null) {
        Log.info('ComicImage', 'Colorization: done ${provider.key}');
        setState(() {
          _colorizedBytes = result;
          _isColorizing = false;
        });
      } else {
        Log.warning('ComicImage', 'Colorization: null result for ${provider.key}');
        _isColorizing = false;
      }
    } catch (e, s) {
      Log.error('ComicImage', 'Colorization processing error: $e', s);
      _isColorizing = false;
    }
  }

  /// 从可能被 ResizeImage 包装的 widget.image 中提取 ReaderImageProvider
  ReaderImageProvider? _getReaderImageProvider() {
    ImageProvider imageProvider = widget.image;
    if (imageProvider is ResizeImage) {
      imageProvider = imageProvider.imageProvider;
    }
    return imageProvider is ReaderImageProvider
        ? imageProvider as ReaderImageProvider
        : null;
  }

  void _resolveImage() {
    final ScrollAwareImageProvider provider = ScrollAwareImageProvider<Object>(
      context: _scrollAwareContext,
      imageProvider: widget.image,
    );
    final ImageStream newStream =
        provider.resolve(createLocalImageConfiguration(
      context,
      size: widget.width != null && widget.height != null
          ? Size(widget.width!, widget.height!)
          : null,
    ));
    _updateSourceStream(newStream);
  }

  ImageStreamListener? _imageStreamListener;

  ImageStreamListener _getListener({bool recreateListener = false}) {
    if (_imageStreamListener == null || recreateListener) {
      _lastException = null;
      _imageStreamListener = ImageStreamListener(
        _handleImageFrame,
        onChunk: _handleImageChunk,
        onError: (Object error, StackTrace? stackTrace) {
          setState(() {
            _lastException = error;
          });
        },
      );
    }
    return _imageStreamListener!;
  }

  void _handleImageFrame(ImageInfo imageInfo, bool synchronousCall) {
    setState(() {
      _replaceImage(info: imageInfo);
      _loadingProgress = null;
      _lastException = null;
      _frameNumber = _frameNumber == null ? 0 : _frameNumber! + 1;
      _wasSynchronouslyLoaded = _wasSynchronouslyLoaded | synchronousCall;
    });
  }

  void _handleImageChunk(ImageChunkEvent event) {
    setState(() {
      _loadingProgress = event;
      _lastException = null;
    });
  }

  void _replaceImage({required ImageInfo? info}) {
    final ImageInfo? oldImageInfo = _imageInfo;
    SchedulerBinding.instance
        .addPostFrameCallback((_) => oldImageInfo?.dispose());
    _imageInfo = info;
  }

  // Updates _imageStream to newStream, and moves the stream listener
  // registration from the old stream to the new stream (if a listener was
  // registered).
  void _updateSourceStream(ImageStream newStream) {
    if (_imageStream?.key == newStream.key) {
      return;
    }

    if (_isListeningToStream) {
      _imageStream!.removeListener(_getListener());
    }

    if (!widget.gaplessPlayback) {
      setState(() {
        _replaceImage(info: null);
      });
    }

    setState(() {
      _loadingProgress = null;
      _frameNumber = null;
      _wasSynchronouslyLoaded = false;
    });

    _imageStream = newStream;
    if (_isListeningToStream) {
      _imageStream!.addListener(_getListener());
    }
  }

  void _listenToStream() {
    if (_isListeningToStream) {
      return;
    }

    _imageStream!.addListener(_getListener());
    _completerHandle?.dispose();
    _completerHandle = null;

    _isListeningToStream = true;
  }

  /// Stops listening to the image stream, if this state object has attached a
  /// listener.
  ///
  /// If the listener from this state is the last listener on the stream, the
  /// stream will be disposed. To keep the stream alive, set `keepStreamAlive`
  /// to true, which create [ImageStreamCompleterHandle] to keep the completer
  /// alive and is compatible with the [TickerMode] being off.
  void _stopListeningToStream({bool keepStreamAlive = false}) {
    if (!_isListeningToStream) {
      return;
    }

    if (keepStreamAlive &&
        _completerHandle == null &&
        _imageStream?.completer != null) {
      _completerHandle = _imageStream!.completer!.keepAlive();
    }

    _imageStream!.removeListener(_getListener());
    _isListeningToStream = false;
  }

  @override
  Widget build(BuildContext context) {
    if (_lastException != null) {
      // display error and retry button on screen
      return SizedBox(
        height: widget.height == null ? 300 : null,
        width: widget.width == null ? 300 : null,
        child: Center(
          child: SizedBox(
            height: 300,
            child: Column(
              children: [
                Expanded(
                  child: Center(
                    child: Text(
                      _lastException.toString(),
                      maxLines: 3,
                    ),
                  ),
                ),
                const SizedBox(
                  height: 4,
                ),
                MouseRegion(
                  cursor: SystemMouseCursors.click,
                  child: Listener(
                    onPointerDown: (details) {
                      GlobalState.find<_ReaderGestureDetectorState>().ignoreNextTap();
                      setState(() {
                        _loadingProgress = null;
                        _lastException = null;
                      });
                      _resolveImage();
                    },
                    child: SizedBox(
                      width: 84,
                      height: 36,
                      child: Center(
                        child: Text(
                          "Retry".tl,
                          style: TextStyle(color: Colors.blue),
                        ),
                      ),
                    ),
                  ),
                ),
                const SizedBox(
                  height: 16,
                ),
              ],
            ),
          ),
        ),
      );
    }

    return LayoutBuilder(builder: (context, constrains) {
      var width = widget.width;
      var height = widget.height;

      if (_imageInfo != null) {
        // Record the height and the width of the image
        _cache[widget.image.hashCode] = Size(_imageInfo!.image.width.toDouble(),
            _imageInfo!.image.height.toDouble());
      }

      Size? cacheSize = _cache[widget.image.hashCode];
      if (cacheSize != null) {
        if (width == double.infinity) {
          width = constrains.maxWidth;
          height = width * cacheSize.height / cacheSize.width;
        } else if (height == double.infinity) {
          height = constrains.maxHeight;
          width = height * cacheSize.width / cacheSize.height;
        }
      } else {
        if (width == double.infinity) {
          width = constrains.maxWidth;
          height = 300;
        } else if (height == double.infinity) {
          height = constrains.maxHeight;
          width = 300;
        }
      }

      if (_colorizedBytes != null && _imageInfo != null) {
        // 使用上色后的图像
        Widget result = Image.memory(
          _colorizedBytes!,
          width: width,
          height: height,
          fit: widget.fit ?? BoxFit.contain,
          filterQuality: widget.filterQuality,
          alignment: widget.alignment,
          repeat: widget.repeat,
        );
        if (!widget.excludeFromSemantics) {
          result = Semantics(
            container: widget.semanticLabel != null,
            image: true,
            label: widget.semanticLabel ?? '',
            child: result,
          );
        }
        result = SizedBox(
          width: width,
          height: height,
          child: Center(
            child: result,
          ),
        );
        return result;
      } else if (_upscaledBytes != null && _imageInfo != null) {
        // 使用超分后的图像
        Widget result = Image.memory(
          _upscaledBytes!,
          width: width,
          height: height,
          fit: widget.fit ?? BoxFit.contain,
          filterQuality: widget.filterQuality,
          alignment: widget.alignment,
          repeat: widget.repeat,
        );
        if (!widget.excludeFromSemantics) {
          result = Semantics(
            container: widget.semanticLabel != null,
            image: true,
            label: widget.semanticLabel ?? '',
            child: result,
          );
        }
        result = SizedBox(
          width: width,
          height: height,
          child: Center(
            child: result,
          ),
        );
        return result;
      } else if (_imageInfo != null) {
        // build image
        Widget result = RawImage(
          // Do not clone the image, because RawImage is a stateless wrapper.
          // The image will be disposed by this state object when it is not needed
          // anymore, such as when it is unmounted or when the image stream pushes
          // a new image.
          image: _imageInfo?.image,
          debugImageLabel: _imageInfo?.debugLabel,
          width: width,
          height: height,
          scale: _imageInfo?.scale ?? 1.0,
          color: widget.color,
          opacity: widget.opacity,
          colorBlendMode: widget.colorBlendMode,
          fit: widget.fit,
          alignment: widget.alignment,
          repeat: widget.repeat,
          centerSlice: widget.centerSlice,
          matchTextDirection: widget.matchTextDirection,
          invertColors: _invertColors,
          isAntiAlias: widget.isAntiAlias,
          filterQuality: widget.filterQuality,
        );

        if (!widget.excludeFromSemantics) {
          result = Semantics(
            container: widget.semanticLabel != null,
            image: true,
            label: widget.semanticLabel ?? '',
            child: result,
          );
        }
        result = SizedBox(
          width: width,
          height: height,
          child: Center(
            child: result,
          ),
        );
        return result;
      } else {
        // build progress
        return SizedBox(
          width: width,
          height: height,
          child: Center(
            child: SizedBox(
              width: 24,
              height: 24,
              child: CircularProgressIndicator(
                strokeWidth: 3,
                backgroundColor: context.colorScheme.surfaceContainer,
                value: (_loadingProgress != null &&
                        _loadingProgress!.expectedTotalBytes != null &&
                        _loadingProgress!.expectedTotalBytes! != 0)
                    ? _loadingProgress!.cumulativeBytesLoaded /
                        _loadingProgress!.expectedTotalBytes!
                    : 0,
              ),
            ),
          ),
        );
      }
    });
  }

  @override
  void debugFillProperties(DiagnosticPropertiesBuilder description) {
    super.debugFillProperties(description);
    description.add(DiagnosticsProperty<ImageStream>('stream', _imageStream));
    description.add(DiagnosticsProperty<ImageInfo>('pixels', _imageInfo));
    description.add(DiagnosticsProperty<ImageChunkEvent>(
        'loadingProgress', _loadingProgress));
    description.add(DiagnosticsProperty<int>('frameNumber', _frameNumber));
    description.add(DiagnosticsProperty<bool>(
        'wasSynchronouslyLoaded', _wasSynchronouslyLoaded));
  }
}
