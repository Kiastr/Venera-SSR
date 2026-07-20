import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

/// Anime4K 超分辨率算法实现
///
/// 基于 Anime4K v1.0 的 "Push Pixels" 算法，
/// 通过梯度上升将像素推向边缘，实现动漫/漫画图像的实时高质量放大。
///
/// 算法步骤：
/// 1. 双线性插值放大图像到目标分辨率
/// 2. 计算亮度图
/// 3. 线条细化（Unblur）：将暗像素推向亮区域以细化线条
/// 4. 计算 Sobel 梯度
/// 5. 梯度精炼（Gradient Refine）：利用梯度信息将像素推向边缘
class Anime4KUpscaler {
  /// 边缘保护阈值 (0-255)：Sobel 梯度幅值高于此值的像素视为"清晰边/细线"，
  /// Unblur 步骤会跳过它们，避免把漫画细线/排线侵蚀掉（核心质量修复）。
  static const int _kEdgeProtectThreshold = 100;

  /// 推送强度 (0.0 - 1.0)，控制像素推送力度
  final double pushStrength;

  /// 梯度精炼强度 (0.0 - 1.0)
  final double pushGradStrength;

  /// 放大倍数
  final double scaleFactor;

  Anime4KUpscaler({
    this.pushStrength = 0.31,
    this.pushGradStrength = 1.0,
    this.scaleFactor = 2.0,
  });

  /// 在 Isolate 中处理图像超分
  /// [params] Anime4K 处理参数
  /// 返回处理后的图像字节数据 (PNG 格式)
  static Future<Uint8List?> processInIsolate(Anime4KParams params) async {
    try {
      return await compute(_processImage, params);
    } catch (e, s) {
      debugPrint('Anime4K processing error: $e\n$s');
      // Fallback: process in the current isolate so that transient isolate
      // failures (e.g. on constrained CI runners) do not silently disable
      // the feature.
      return _processImage(params);
    }
  }

  /// 在 Isolate 中执行的静态方法
  static Uint8List? _processImage(Anime4KParams params) {
    try {
      final upscaler = Anime4KUpscaler(
        pushStrength: params.pushStrength,
        pushGradStrength: params.pushGradStrength,
        scaleFactor: params.scaleFactor,
      );

      final srcImage = img.decodeImage(params.imageBytes);
      if (srcImage == null) {
        debugPrint('Anime4K: failed to decode image bytes');
        return null;
      }

      final result = upscaler.upscale(srcImage);
      return Uint8List.fromList(img.encodePng(result));
    } catch (e, s) {
      debugPrint('Anime4K upscale error: $e\n$s');
      return null;
    }
  }

  /// 直接在当前 Isolate 中处理图像（用于测试或作为 compute 失败时的回退）
  static Uint8List? processDirect(Anime4KParams params) {
    return _processImage(params);
  }

  /// 对图像执行 Anime4K 超分处理
  img.Image upscale(img.Image source) {
    // 步骤1: 双线性插值放大
    final int newWidth = (source.width * scaleFactor).round();
    final int newHeight = (source.height * scaleFactor).round();

    final img.Image upscaled = img.copyResize(
      source,
      width: newWidth,
      height: newHeight,
      interpolation: img.Interpolation.linear,
    );

    final int width = upscaled.width;
    final int height = upscaled.height;
    final int size = width * height;

    // 提取颜色数据到平面数组
    final Int32List colorData = Int32List(size);
    final Uint8List lumData = Uint8List(size);
    final Int32List tempColorData = Int32List(size);
    final Uint8List tempLumData = Uint8List(size);

    // 填充颜色数据
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final pixel = upscaled.getPixel(x, y);
        final int idx = y * width + x;
        colorData[idx] = _toARGB(
          pixel.a.toInt(),
          pixel.r.toInt(),
          pixel.g.toInt(),
          pixel.b.toInt(),
        );
      }
    }

    // 步骤2: 计算亮度
    _computeLuminance(colorData, lumData, size);

    // 步骤2.5: 计算边缘掩码（Sobel 梯度幅值），用于保护清晰边/细线不被 Unblur 侵蚀
    final Uint8List edgeMask = _computeEdgeMask(lumData, width, height);

    // 步骤3: 线条细化 (Unblur)
    final int unblurStrength = (pushStrength * 255).round().clamp(0, 0xFFFF);
    int remaining = unblurStrength;
    bool forward = true;

    while (remaining > 0) {
      final int current = remaining.clamp(0, 255);
      if (forward) {
        _unblur(colorData, lumData, tempColorData, tempLumData, width, height,
            current, edgeMask);
      } else {
        _unblur(tempColorData, tempLumData, colorData, lumData, width, height,
            current, edgeMask);
      }
      forward = !forward;
      remaining -= current;
    }

    // 确保数据在 colorData/lumData 中
    if (!forward) {
      colorData.setAll(0, tempColorData);
      lumData.setAll(0, tempLumData);
    }

    // 步骤4: 计算 Sobel 梯度
    _computeGradient(colorData, lumData, tempColorData, tempLumData, width,
        height);
    // 结果在 tempColorData/tempLumData 中
    colorData.setAll(0, tempColorData);
    lumData.setAll(0, tempLumData);

    // 步骤5: 梯度精炼
    final int refineStrength =
        (pushGradStrength * 255).round().clamp(0, 0xFFFF);
    remaining = refineStrength;
    forward = true;

    while (remaining > 0) {
      final int current = remaining.clamp(0, 255);
      if (forward) {
        _gradientRefine(colorData, lumData, tempColorData, tempLumData, width,
            height, current);
      } else {
        _gradientRefine(tempColorData, tempLumData, colorData, lumData, width,
            height, current);
      }
      forward = !forward;
      remaining -= current;
    }

    // 确保数据在 colorData 中
    if (!forward) {
      colorData.setAll(0, tempColorData);
    }

    // 写回图像
    final img.Image result =
        img.Image(width: width, height: height, numChannels: 4);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int idx = y * width + x;
        final int argb = colorData[idx];
        result.setPixelRgba(
          x,
          y,
          _getRed(argb),
          _getGreen(argb),
          _getBlue(argb),
          _getAlpha(argb),
        );
      }
    }

    return result;
  }

  // ============ 核心算法函数 ============

  /// 计算亮度
  void _computeLuminance(Int32List colorData, Uint8List lumData, int size) {
    for (int i = 0; i < size; i++) {
      final int rgb = colorData[i];
      final int r = _getRed(rgb);
      final int g = _getGreen(rgb);
      final int b = _getBlue(rgb);
      lumData[i] = _getLuminance(r, g, b);
    }
  }

  /// 计算 Sobel 梯度幅值掩码（0-255），用于在 Unblur 前识别清晰边/细线。
  /// 梯度越高的像素越可能是线条或锐边，应被保护以免被 Unblur 侵蚀。
  Uint8List _computeEdgeMask(Uint8List lumData, int width, int height) {
    final Uint8List mask = Uint8List(width * height);
    for (int y = 0; y < height; y++) {
      final int yn = y == 0 ? 0 : -width;
      final int yp = y == height - 1 ? 0 : width;
      for (int x = 0; x < width; x++) {
        final int id = y * width + x;
        final int xn = x == 0 ? 0 : -1;
        final int xp = x == width - 1 ? 0 : 1;

        final int tl = lumData[id + yn + xn];
        final int t = lumData[id + yn];
        final int tr = lumData[id + yn + xp];
        final int l = lumData[id + xn];
        final int r = lumData[id + xp];
        final int bl = lumData[id + yp + xn];
        final int b = lumData[id + yp];
        final int br = lumData[id + yp + xp];

        final int xSobel =
            (-tl + tr - l - l + r + r - bl + br).abs();
        final int ySobel =
            (-tl - t - t - tr + bl + b + b + br).abs();
        mask[id] = math
            .sqrt(xSobel * xSobel + ySobel * ySobel)
            .round()
            .clamp(0, 255);
      }
    }
    return mask;
  }

  /// 线条细化 (Unblur) - 将暗像素推向亮区域
  ///
  /// [edgeMask] Sobel 梯度幅值掩码（0-255）。梯度高于 [_kEdgeProtectThreshold]
  /// 的像素（清晰边/细线）直接原样保留，避免 Unblur 把漫画细线侵蚀掉。
  void _unblur(
    Int32List srcColor,
    Uint8List srcLum,
    Int32List dstColor,
    Uint8List dstLum,
    int width,
    int height,
    int strength,
    Uint8List edgeMask,
  ) {
    strength = strength.clamp(0, 255);

    for (int y = 0; y < height; y++) {
      final int yn = y == 0 ? 0 : -width;
      final int yp = y == height - 1 ? 0 : width;

      for (int x = 0; x < width; x++) {
        final int id = y * width + x;

        // 边缘保护：清晰边/细线（高梯度）原样保留，防止细线被侵蚀
        if (edgeMask[id] >= _kEdgeProtectThreshold) {
          dstColor[id] = srcColor[id];
          dstLum[id] = srcLum[id];
          continue;
        }

        final int xn = x == 0 ? 0 : -1;
        final int xp = x == width - 1 ? 0 : 1;

        // 邻域索引
        final int ti = id + yn;
        final int tli = ti + xn;
        final int tri = ti + xp;
        final int li = id + xn;
        final int ri = id + xp;
        final int bi = id + yp;
        final int bli = bi + xn;
        final int bri = bi + xp;

        int lightestColor = srcColor[id];
        int lightestLum = srcLum[id];

        // 检查8个方向的核
        for (int k = 0; k < 8; k++) {
          late int di0, di1, di2;
          late int li0, li1, li2;
          int li3 = id;
          bool l4 = false;

          switch (k) {
            case 0:
              di0 = tli; di1 = ti; di2 = tri;
              li0 = id; li1 = bli; li2 = bi; li3 = bri; l4 = true;
              break;
            case 1:
              di0 = ti; di1 = tri; di2 = ri;
              li0 = id; li1 = li; li2 = bi; l4 = false;
              break;
            case 2:
              di0 = tri; di1 = ri; di2 = bri;
              li0 = id; li1 = tli; li2 = li; li3 = bli; l4 = true;
              break;
            case 3:
              di0 = ri; di1 = bri; di2 = bi;
              li0 = id; li1 = ti; li2 = li; l4 = false;
              break;
            case 4:
              di0 = bli; di1 = bi; di2 = bri;
              li0 = id; li1 = tli; li2 = ti; li3 = tri; l4 = true;
              break;
            case 5:
              di0 = li; di1 = bli; di2 = bi;
              li0 = id; li1 = ti; li2 = ri; l4 = false;
              break;
            case 6:
              di0 = tli; di1 = li; di2 = bli;
              li0 = id; li1 = tri; li2 = ri; li3 = bri; l4 = true;
              break;
            case 7:
              di0 = tli; di1 = ti; di2 = li;
              li0 = id; li1 = bi; li2 = ri; l4 = false;
              break;
          }

          final int d0 = srcLum[di0];
          final int d1 = srcLum[di1];
          final int d2 = srcLum[di2];
          final int lv0 = srcLum[li0];
          final int lv1 = srcLum[li1];
          final int lv2 = srcLum[li2];

          bool match;
          if (l4) {
            final int lv3 = srcLum[li3];
            match = !_compareLum4(d0, d1, d2, lv0, lv1, lv2, lv3);
          } else {
            match = !_compareLum3(d0, d1, d2, lv0, lv1, lv2);
          }

          if (match) {
            final int newColor = _weightedAverageRGB(
              srcColor[id],
              _averageRGB(srcColor[di0], srcColor[di1], srcColor[di2]),
              strength,
            );
            final int newLum = _getLuminance(
              _getRed(newColor),
              _getGreen(newColor),
              _getBlue(newColor),
            );
            if (newLum > lightestLum) {
              lightestLum = newLum;
              lightestColor = newColor;
            }
          }
        }

        dstColor[id] = lightestColor;
        dstLum[id] = lightestLum.clamp(0, 255);
      }
    }
  }

  /// 计算 Sobel 梯度
  void _computeGradient(
    Int32List srcColor,
    Uint8List srcLum,
    Int32List dstColor,
    Uint8List dstLum,
    int width,
    int height,
  ) {
    for (int y = 0; y < height; y++) {
      final int yn = y == 0 ? 0 : -width;
      final int yp = y == height - 1 ? 0 : width;

      for (int x = 0; x < width; x++) {
        final int id = y * width + x;
        final int xn = x == 0 ? 0 : -1;
        final int xp = x == width - 1 ? 0 : 1;

        final int topi = id + yn;
        final int topLefti = topi + xn;
        final int topRighti = topi + xp;
        final int lefti = id + xn;
        final int righti = id + xp;
        final int bottomi = id + yp;
        final int bottomLefti = bottomi + xn;
        final int bottomRighti = bottomi + xp;

        final int topLeft = srcLum[topLefti];
        final int top = srcLum[topi];
        final int topRight = srcLum[topRighti];
        final int left = srcLum[lefti];
        final int right = srcLum[righti];
        final int bottomLeft = srcLum[bottomLefti];
        final int bottom = srcLum[bottomi];
        final int bottomRight = srcLum[bottomRighti];

        // Sobel 算子
        final int xSobel =
            (-topLeft + topRight - left - left + right + right - bottomLeft +
                    bottomRight)
                .abs();
        final int ySobel =
            (-topLeft - top - top - topRight + bottomLeft + bottom + bottom +
                    bottomRight)
                .abs();

        final int deriv = math.sqrt(xSobel * xSobel + ySobel * ySobel)
            .round()
            .clamp(0, 255);

        dstLum[id] = deriv;
        dstColor[id] = srcColor[id];
      }
    }
  }

  /// 梯度精炼 - 将像素推向边缘
  void _gradientRefine(
    Int32List srcColor,
    Uint8List srcLum,
    Int32List dstColor,
    Uint8List dstLum,
    int width,
    int height,
    int strength,
  ) {
    strength = strength.clamp(0, 255);

    for (int y = 0; y < height; y++) {
      final int yn = y == 0 ? 0 : -width;
      final int yp = y == height - 1 ? 0 : width;

      for (int x = 0; x < width; x++) {
        final int id = y * width + x;
        final int xn = x == 0 ? 0 : -1;
        final int xp = x == width - 1 ? 0 : 1;

        final int ti = id + yn;
        final int tli = ti + xn;
        final int tri = ti + xp;
        final int li = id + xn;
        final int ri = id + xp;
        final int bi = id + yp;
        final int bli = bi + xn;
        final int bri = bi + xp;

        bool pushed = false;

        for (int k = 0; k < 8; k++) {
          late int di0, di1, di2;
          late int lvi0, lvi1, lvi2;
          int lvi3 = id;
          bool l4 = false;

          switch (k) {
            case 0:
              di0 = tli; di1 = ti; di2 = tri;
              lvi0 = id; lvi1 = bli; lvi2 = bi; lvi3 = bri; l4 = true;
              break;
            case 1:
              di0 = ti; di1 = tri; di2 = ri;
              lvi0 = id; lvi1 = li; lvi2 = bi; l4 = false;
              break;
            case 2:
              di0 = tri; di1 = ri; di2 = bri;
              lvi0 = id; lvi1 = tli; lvi2 = li; lvi3 = bli; l4 = true;
              break;
            case 3:
              di0 = ri; di1 = bri; di2 = bi;
              lvi0 = id; lvi1 = ti; lvi2 = li; l4 = false;
              break;
            case 4:
              di0 = bli; di1 = bi; di2 = bri;
              lvi0 = id; lvi1 = tli; lvi2 = ti; lvi3 = tri; l4 = true;
              break;
            case 5:
              di0 = li; di1 = bli; di2 = bi;
              lvi0 = id; lvi1 = ti; lvi2 = ri; l4 = false;
              break;
            case 6:
              di0 = tli; di1 = li; di2 = bli;
              lvi0 = id; lvi1 = tri; lvi2 = ri; lvi3 = bri; l4 = true;
              break;
            case 7:
              di0 = tli; di1 = ti; di2 = li;
              lvi0 = id; lvi1 = bi; lvi2 = ri; l4 = false;
              break;
          }

          final int d0 = srcLum[di0];
          final int d1 = srcLum[di1];
          final int d2 = srcLum[di2];
          final int lv0 = srcLum[lvi0];
          final int lv1 = srcLum[lvi1];
          final int lv2 = srcLum[lvi2];

          bool match;
          if (l4) {
            final int lv3 = srcLum[lvi3];
            match = _compareLum4(d0, d1, d2, lv0, lv1, lv2, lv3);
          } else {
            match = _compareLum3(d0, d1, d2, lv0, lv1, lv2);
          }

          if (match) {
            dstLum[id] = _weightedAverageGray(
              srcLum[id],
              _averageGray(d0, d1, d2),
              strength,
            );
            dstColor[id] = _weightedAverageRGB(
              srcColor[id],
              _averageRGB(srcColor[di0], srcColor[di1], srcColor[di2]),
              strength,
            );
            pushed = true;
            break;
          }
        }

        if (!pushed) {
          dstLum[id] = srcLum[id];
          dstColor[id] = srcColor[id];
        }
      }
    }
  }

  // ============ 辅助函数 ============

  static int _getAlpha(int argb) => (argb >> 24) & 0xFF;
  static int _getRed(int argb) => (argb >> 16) & 0xFF;
  static int _getGreen(int argb) => (argb >> 8) & 0xFF;
  static int _getBlue(int argb) => argb & 0xFF;

  static int _toARGB(int a, int r, int g, int b) {
    return ((a.clamp(0, 255) << 24) |
        (r.clamp(0, 255) << 16) |
        (g.clamp(0, 255) << 8) |
        b.clamp(0, 255));
  }

  static int _getLuminance(int r, int g, int b) {
    return ((r + r + g + g + g + b) ~/ 6).clamp(0, 255);
  }

  /// 比较3个暗像素是否都小于3个亮像素
  static bool _compareLum3(
      int d0, int d1, int d2, int l0, int l1, int l2) {
    return d0 < l0 &&
        d0 < l1 &&
        d0 < l2 &&
        d1 < l0 &&
        d1 < l1 &&
        d1 < l2 &&
        d2 < l0 &&
        d2 < l1 &&
        d2 < l2;
  }

  /// 比较3个暗像素是否都小于4个亮像素
  static bool _compareLum4(
      int d0, int d1, int d2, int l0, int l1, int l2, int l3) {
    return d0 < l0 &&
        d0 < l1 &&
        d0 < l2 &&
        d0 < l3 &&
        d1 < l0 &&
        d1 < l1 &&
        d1 < l2 &&
        d1 < l3 &&
        d2 < l0 &&
        d2 < l1 &&
        d2 < l2 &&
        d2 < l3;
  }

  static int _averageGray(int d0, int d1, int d2) {
    return ((d0 + d1 + d2) ~/ 3).clamp(0, 255);
  }

  static int _averageRGB(int c0, int c1, int c2) {
    final int ra = (_getRed(c0) + _getRed(c1) + _getRed(c2)) ~/ 3;
    final int ga = (_getGreen(c0) + _getGreen(c1) + _getGreen(c2)) ~/ 3;
    final int ba = (_getBlue(c0) + _getBlue(c1) + _getBlue(c2)) ~/ 3;
    final int aa = (_getAlpha(c0) + _getAlpha(c1) + _getAlpha(c2)) ~/ 3;
    return _toARGB(aa, ra, ga, ba);
  }

  static int _weightedAverageGray(int d0, int d1, int alpha) {
    return ((d0 * (255 - alpha) + d1 * alpha) ~/ 255).clamp(0, 255);
  }

  static int _weightedAverageRGB(int c0, int c1, int alpha) {
    final int ra =
        (_getRed(c0) * (255 - alpha) + _getRed(c1) * alpha) ~/ 255;
    final int ga =
        (_getGreen(c0) * (255 - alpha) + _getGreen(c1) * alpha) ~/ 255;
    final int ba =
        (_getBlue(c0) * (255 - alpha) + _getBlue(c1) * alpha) ~/ 255;
    final int aa =
        (_getAlpha(c0) * (255 - alpha) + _getAlpha(c1) * alpha) ~/ 255;
    return _toARGB(aa, ra, ga, ba);
  }
}

/// Anime4K 处理参数（用于 Isolate 传递）
class Anime4KParams {
  final Uint8List imageBytes;
  final double pushStrength;
  final double pushGradStrength;
  final double scaleFactor;

  const Anime4KParams({
    required this.imageBytes,
    this.pushStrength = 0.31,
    this.pushGradStrength = 1.0,
    this.scaleFactor = 2.0,
  });
}
