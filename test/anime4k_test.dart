import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;

import 'package:venera/utils/anime4k/anime4k_upscaler.dart';

/// Tests for the Anime4K super-resolution component.
///
/// These tests verify the pure-Dart Anime4K implementation without relying on
/// Flutter widgets, ONNX models, or isolate infrastructure. They focus on the
/// core algorithm contract: given valid image bytes, the upscaler must produce
/// a larger, decodable PNG whose dimensions match the requested scale factor.
void main() {
  group('Anime4KUpscaler.processDirect', () {
    test('upscales a small RGB image by the given scale factor', () {
      final src = img.Image(width: 8, height: 8, numChannels: 4);
      // Draw a simple pattern: left half dark, right half bright.
      for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
          if (x < 4) {
            src.setPixelRgba(x, y, 20, 20, 20, 255);
          } else {
            src.setPixelRgba(x, y, 230, 230, 230, 255);
          }
        }
      }

      final params = Anime4KParams(
        imageBytes: Uint8List.fromList(img.encodePng(src)),
        scaleFactor: 2.0,
        pushStrength: 0.31,
        pushGradStrength: 1.0,
      );

      final result = Anime4KUpscaler.processDirect(params);

      expect(result, isNotNull);
      expect(result!.isNotEmpty, true);

      final decoded = img.decodeImage(result);
      expect(decoded, isNotNull);
      // 8 * 2.0 = 16
      expect(decoded!.width, 16);
      expect(decoded.height, 16);
    });

    test('produces a valid PNG signature', () {
      final src = img.Image(width: 4, height: 4, numChannels: 4);
      img.fill(src, color: img.ColorRgba8(128, 128, 128, 255));

      final params = Anime4KParams(
        imageBytes: Uint8List.fromList(img.encodePng(src)),
        scaleFactor: 1.5,
      );

      final result = Anime4KUpscaler.processDirect(params);
      expect(result, isNotNull);
      // PNG signature: 137 80 78 71 13 10 26 10
      expect(result![0], 0x89);
      expect(result[1], 0x50); // 'P'
      expect(result[2], 0x4E); // 'N'
      expect(result[3], 0x47); // 'G'
    });

    test('returns null for invalid image bytes', () {
      final params = Anime4KParams(
        imageBytes: Uint8List.fromList([0, 1, 2, 3, 4]),
        scaleFactor: 2.0,
      );

      final result = Anime4KUpscaler.processDirect(params);
      expect(result, isNull);
    });

    test('respects scaleFactor parameter (1.0 keeps size)', () {
      final src = img.Image(width: 10, height: 6, numChannels: 4);
      img.fill(src, color: img.ColorRgba8(100, 150, 200, 255));

      final params = Anime4KParams(
        imageBytes: Uint8List.fromList(img.encodePng(src)),
        scaleFactor: 1.0,
      );

      final result = Anime4KUpscaler.processDirect(params);
      expect(result, isNotNull);
      final decoded = img.decodeImage(result!);
      expect(decoded, isNotNull);
      expect(decoded!.width, 10);
      expect(decoded.height, 6);
    });

    test('handles 3x scale factor', () {
      final src = img.Image(width: 5, height: 5, numChannels: 4);
      for (int y = 0; y < 5; y++) {
        for (int x = 0; x < 5; x++) {
          src.setPixelRgba(x, y, x * 50, y * 50, 128, 255);
        }
      }

      final params = Anime4KParams(
        imageBytes: Uint8List.fromList(img.encodePng(src)),
        scaleFactor: 3.0,
      );

      final result = Anime4KUpscaler.processDirect(params);
      expect(result, isNotNull);
      final decoded = img.decodeImage(result!);
      expect(decoded, isNotNull);
      expect(decoded!.width, 15);
      expect(decoded.height, 15);
    });

    test('preserves alpha channel in output', () {
      final src = img.Image(width: 6, height: 6, numChannels: 4);
      // Half transparent, half opaque.
      for (int y = 0; y < 6; y++) {
        for (int x = 0; x < 6; x++) {
          final alpha = x < 3 ? 0 : 255;
          src.setPixelRgba(x, y, 200, 100, 50, alpha);
        }
      }

      final params = Anime4KParams(
        imageBytes: Uint8List.fromList(img.encodePng(src)),
        scaleFactor: 2.0,
      );

      final result = Anime4KUpscaler.processDirect(params);
      expect(result, isNotNull);
      final decoded = img.decodeImage(result!);
      expect(decoded, isNotNull);
      // Output should be 4-channel (RGBA)
      expect(decoded!.numChannels, 4);
    });
  });

  group('Anime4KUpscaler.processInIsolate', () {
    test('falls back to direct processing when isolate fails', () async {
      final src = img.Image(width: 8, height: 8, numChannels: 4);
      img.fill(src, color: img.ColorRgba8(180, 180, 180, 255));

      final params = Anime4KParams(
        imageBytes: Uint8List.fromList(img.encodePng(src)),
        scaleFactor: 2.0,
      );

      // processInIsolate should either succeed via compute() or fall back
      // to _processImage in the current isolate. Either way, the result
      // must be non-null for valid input.
      final result = await Anime4KUpscaler.processInIsolate(params);
      expect(result, isNotNull);
      expect(result!.isNotEmpty, true);

      final decoded = img.decodeImage(result);
      expect(decoded, isNotNull);
      expect(decoded!.width, 16);
      expect(decoded.height, 16);
    });
  });
}
