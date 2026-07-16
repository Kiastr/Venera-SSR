import 'dart:async' show Future;
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_qjs/flutter_qjs.dart';
import 'package:venera/foundation/js_engine.dart';
import 'package:venera/network/images.dart';
import 'package:venera/utils/io.dart';
import 'package:venera/foundation/log.dart';
import 'base_image_provider.dart';
import 'reader_image.dart' as image_provider;
import 'package:venera/foundation/appdata.dart';
import 'package:venera/utils/anime4k/anime4k_service.dart';
import 'package:venera/utils/colorization/colorization_service.dart';

class ReaderImageProvider
    extends BaseImageProvider<image_provider.ReaderImageProvider> {
  /// Image provider for normal image.
  const ReaderImageProvider(this.imageKey, this.sourceKey, this.cid, this.eid, this.page);

  final String imageKey;

  final String? sourceKey;

  final String cid;

  final String eid;

  final int page;

  @override
  Future<Uint8List> load(chunkEvents, checkStop) async {
    Uint8List? imageBytes;
    if (imageKey.startsWith('file://')) {
      // Strip the "file://" prefix to get the actual file path.
      // LocalManager stores local image keys as "file://<absolutePath>",
      // so we must remove the scheme before constructing a [File].
      var file = File(imageKey.substring(7));
      if (await file.exists()) {
        imageBytes = await file.readAsBytes();
      } else {
        throw "Error: File not found.";
      }
    } else {
      await for (var event
        in ImageDownloader.loadComicImage(imageKey, sourceKey, cid, eid)) {
        checkStop();
        chunkEvents.add(ImageChunkEvent(
          cumulativeBytesLoaded: event.currentBytes,
          expectedTotalBytes: event.totalBytes,
        ));
        if (event.imageBytes != null) {
          imageBytes = event.imageBytes;
          break;
        }
      }
    }
    if (imageBytes == null) {
      throw "Error: Empty response body.";
    }
    // 自此 imageBytes 非 null，用 final 捕获以便下方闭包/赋值使用
    var bytes = imageBytes;
    if (appdata.settings['enableCustomImageProcessing']) {
      var script = appdata.settings['customImageProcessing'].toString();
      if (!script.contains('function processImage')) {
        return bytes;
      }
      var func = JsEngine().runCode('''
        (() => {
          $script
          return processImage;
        })()
      ''');
      if (func is JSInvokable) {
        var autoFreeFunc = JSAutoFreeFunction(func);
        var result = autoFreeFunc([bytes, cid, eid, page, sourceKey]);
        if (result is Uint8List) {
          bytes = result;
        } else if (result is Future) {
          var futureResult = await result;
          if (futureResult is Uint8List) {
            bytes = futureResult;
          }
        } else if (result is Map) {
          var image = result['image'];
          if (image is Uint8List) {
            bytes = image;
          } else if (image is Future) {
            JSAutoFreeFunction? onCancel;
            if (result['onCancel'] is JSInvokable) {
              onCancel = JSAutoFreeFunction(result['onCancel']);
            }
            if (onCancel == null) {
              var futureImage = await image;
              if (futureImage is Uint8List) {
                bytes = futureImage;
              }
            } else {
              dynamic futureImage;
              image.then((value) {
                futureImage = value;
                futureImage ??= Uint8List(0);
              });
              while (futureImage == null) {
                try {
                  checkStop();
                }
                catch(e) {
                  onCancel([]);
                  rethrow;
                }
                await Future.delayed(Duration(milliseconds: 50));
              }
              if (futureImage is Uint8List) {
                bytes = futureImage;
              }
            }
          }
        }
      }
    }
    // ===== Anime4K 超分处理 =====
    // 在 ImageProvider.load 阶段处理图片字节，与自定义图片处理相同位置，
    // 确保无论阅读器用什么 widget 渲染都会生效。
    final enableAnime4K = appdata.settings.getReaderSetting(
          cid, sourceKey ?? "", 'enableAnime4K') ==
        true;
    if (enableAnime4K) {
      try {
        final result = await Anime4KService.instance.processImage(
          imageBytes: bytes,
          cacheKey: key,
          scaleFactor: (appdata.settings.getReaderSetting(
                    cid, sourceKey ?? "", 'anime4KScaleFactor') as num?)
                ?.toDouble() ??
              2.0,
          pushStrength: (appdata.settings.getReaderSetting(
                    cid, sourceKey ?? "", 'anime4KPushStrength') as num?)
                ?.toDouble() ??
              0.31,
          pushGradStrength: (appdata.settings.getReaderSetting(
                    cid, sourceKey ?? "", 'anime4KPushGradStrength') as num?)
                ?.toDouble() ??
              1.0,
        );
        if (result != null) {
          bytes = result;
        }
      } catch (e, s) {
        Log.error('ReaderImage', 'Anime4K processing error: $e', s);
      }
    }

    // ===== AI 上色处理 =====
    final enableColorization = appdata.settings.getReaderSetting(
          cid, sourceKey ?? "", 'enableColorization') ==
        true;
    if (enableColorization) {
      try {
        if (!ColorizationService.instance.isModelAvailable) {
          await ColorizationService.instance.checkModelAvailable();
        }
        if (ColorizationService.instance.isModelAvailable) {
          final result = await ColorizationService.instance.processImage(
            imageBytes: bytes,
            cacheKey: key,
            intensity: (appdata.settings.getReaderSetting(
                      cid, sourceKey ?? "", 'colorizationIntensity') as num?)
                  ?.toDouble() ??
                1.0,
          );
          if (result != null) {
            bytes = result;
          }
        }
      } catch (e, s) {
        Log.error('ReaderImage', 'Colorization processing error: $e', s);
      }
    }

    return bytes;
  }

  @override
  Future<ReaderImageProvider> obtainKey(ImageConfiguration configuration) {
    return SynchronousFuture(this);
  }

  @override
  String get key => "$imageKey@$sourceKey@$cid@$eid";

  @override
  bool get enableResize => false;
}
