import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;

import 'package:venera/utils/colorization/colorization_processor.dart';

/// 图像上色后处理单元测试。
///
/// 由于 ONNX 模型文件过大，CI/测试环境不便于下载真实模型，
/// 这里重点验证与官方 deoldify-onnx 实现对齐的纯 Dart 后处理逻辑：
/// - OpenCV 风格 RGB<->LAB 转换
/// - 13x13 高斯模糊（与 cv2.GaussianBlur 等价）
void main() {
  group('rgbToLab / labToRgb', () {
    test('neutral gray maps to L≈gray*2.55, A=B=128', () {
      for (final gray in <int>[0, 64, 128, 192, 255]) {
        final (l, a, b) = rgbToLab(gray.toDouble(), gray.toDouble(), gray.toDouble());
        // 允许 ±1 的舍入误差
        expect(l, closeTo(gray * 2.55, 1.0));
        expect(a, closeTo(128.0, 1.0));
        expect(b, closeTo(128.0, 1.0));
      }
    });

    test('round-trip preserves gray values', () {
      for (final gray in <int>[0, 32, 64, 128, 200, 255]) {
        final (l, a, b) = rgbToLab(gray.toDouble(), gray.toDouble(), gray.toDouble());
        final (r, g, b2) = labToRgb(l, a, b);
        expect(r.round(), closeTo(gray, 1));
        expect(g.round(), closeTo(gray, 1));
        expect(b2.round(), closeTo(gray, 1));
      }
    });

    test('round-trip preserves saturated colors within tolerance', () {
      const colors = <(int, int, int)>[
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (128, 64, 200),
      ];
      for (final (rr, gg, bb) in colors) {
        final (l, a, b) = rgbToLab(rr.toDouble(), gg.toDouble(), bb.toDouble());
        final (r2, g2, b2) = labToRgb(l, a, b);
        expect(r2.round(), closeTo(rr, 2));
        expect(g2.round(), closeTo(gg, 2));
        expect(b2.round(), closeTo(bb, 2));
      }
    });

    test('LAB values stay in OpenCV uint8 ranges', () {
      final (l, a, b) = rgbToLab(255.0, 0.0, 0.0);
      expect(l, inInclusiveRange(0.0, 255.0));
      expect(a, inInclusiveRange(0.0, 255.0));
      expect(b, inInclusiveRange(0.0, 255.0));
    });
  });

  group('gaussianKernel13', () {
    test('has length 13 and sums to 1.0', () {
      final kernel = gaussianKernel13();
      expect(kernel.length, 13);

      double sum = 0.0;
      for (final v in kernel) {
        sum += v;
      }
      expect(sum, closeTo(1.0, 1e-6));
    });

    test('is symmetric around the center', () {
      final kernel = gaussianKernel13();
      for (int i = 0; i < kernel.length; i++) {
        expect(kernel[i], closeTo(kernel[kernel.length - 1 - i], 1e-12));
      }
    });

    test('peak is at the center', () {
      final kernel = gaussianKernel13();
      final center = kernel.length ~/ 2;
      double maxValue = kernel[0];
      int maxIndex = 0;
      for (int i = 1; i < kernel.length; i++) {
        if (kernel[i] > maxValue) {
          maxValue = kernel[i];
          maxIndex = i;
        }
      }
      expect(maxIndex, center);
    });
  });

  group('gaussianBlur13Image', () {
    test('leaves a uniform image unchanged', () {
      final src = img.Image(width: 32, height: 32);
      img.fill(src, color: img.ColorRgb8(120, 130, 140));

      final dst = gaussianBlur13Image(src);

      expect(dst.width, src.width);
      expect(dst.height, src.height);

      for (int y = 0; y < dst.height; y++) {
        for (int x = 0; x < dst.width; x++) {
          final p = dst.getPixel(x, y);
          expect(p.r, closeTo(120, 1));
          expect(p.g, closeTo(130, 1));
          expect(p.b, closeTo(140, 1));
        }
      }
    });

    test('blurs a single bright pixel into neighbors', () {
      final src = img.Image(width: 32, height: 32);
      img.fill(src, color: img.ColorRgb8(0, 0, 0));
      src.setPixelRgb(16, 16, 255, 255, 255);

      final dst = gaussianBlur13Image(src);

      // 中心亮度应显著下降（能量被分散到 13x13 邻域）
      final center = dst.getPixel(16, 16);
      expect(center.r, lessThan(255));
      expect(center.r, greaterThan(0));

      // 邻域应被点亮
      final neighbor = dst.getPixel(17, 17);
      expect(neighbor.r, greaterThan(0));

      // 远距离应保持黑暗
      final far = dst.getPixel(0, 0);
      expect(far.r, equals(0));
    });

    test('preserves image dimensions', () {
      final src = img.Image(width: 7, height: 9);
      final dst = gaussianBlur13Image(src);
      expect(dst.width, 7);
      expect(dst.height, 9);
    });
  });

  group('ColorizationModelManager', () {
    test('model file name is deoldify_artistic.onnx', () {
      expect(ColorizationModelManager.modelFileName, 'deoldify_artistic.onnx');
    });

    test('expected file size is approximately 243MB', () {
      expect(
        ColorizationModelManager.expectedFileSize,
        closeTo(243 * 1024 * 1024, 1024),
      );
    });
  });
}
