# ============================================================
# Venera 图像上色（Colorization）原生依赖混淆保留规则
# release 构建 minifyEnabled=true + shrinkResources=true，
# 必须保留 ONNX Runtime / OpenCV 的 Java 类与 JNI 入口，否则运行期
# 会 NoClassDefFoundError / UnsatisfiedLinkError。
# ============================================================

# ONNX Runtime Android
-keep class ai.onnxruntime.** { *; }
-keep interface ai.onnxruntime.** { *; }
-keep class ai.onnxruntime.providers.** { *; }
-dontwarn ai.onnxruntime.**

# OpenCV (Maven AAR)
-keep class org.opencv.** { *; }
-keep interface org.opencv.** { *; }
-dontwarn org.opencv.**

# 上色插件与方法通道（防止 R8 误删未被直接引用的入口）
-keep class com.github.kiastr.venera_ssr.colorize.** { *; }

# ============================================================
# Flutter embedding（PlayStoreDeferredComponentManager /
# FlutterPlayStoreSplitApplication）引用了 com.google.android.play.core.*，
# 但本项目未启用 deferred-components，Flutter Gradle 插件不会引入 play-core，
# 这些类在运行期也永远不会被走到。用 dontwarn/keep 让 R8 strict 模式放行，
# 等价 R8 自建的 missing_rules.txt 建议（避免 Missing class 报错）。
# ============================================================
-dontwarn com.google.android.play.core.**
-keep class com.google.android.play.core.** { *; }

# 通用：MethodChannel 回调与 Flutter 嵌入层
-keep class io.flutter.** { *; }
-keep class * extends io.flutter.plugin.common.MethodCallHandler { *; }
