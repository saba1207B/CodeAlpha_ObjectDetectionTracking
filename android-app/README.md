# CodeAlpha Android Camera Client (`android-app/`)

This directory contains information and binary builds for the **CodeAlpha Object Detection & Tracking Android Camera Client**.

---

## Pre-Built APK

The ready-to-install debug APK is located at:
```
android-app/app-debug.apk
```
(Also generated at `app/build/outputs/apk/debug/app-debug.apk`).

### Installing on your Android Phone

1. **Via USB (ADB):**
   ```bash
   adb install -r android-app/app-debug.apk
   ```
2. **Via Direct Transfer:**
   - Copy `app-debug.apk` to your phone via USB cable, Google Drive, or local file sharing.
   - On your phone, tap the file in your Files / Downloads app and select **Install**. (Allow "Install Unknown Apps" if prompted).

---

## Source Code Architecture

The Android source code is maintained in standard Android Gradle layout in the root `app/` module to support seamless CI/CD and IDE builds:

- `app/src/main/java/com/example/codealpha_objectdetectiontracking/`:
  - `MainActivity.kt`: User interface with CameraX preview, URL input field, Start/Stop toggle, FPS counter, and pre-flight usage disclaimer notes.
  - `FrameProcessor.kt`: High-efficiency CameraX frame analyzer converting YUV_420_888 camera buffers to JPEG with zero buffer buildup.
  - `FrameSender.kt`: High-performance asynchronous HTTP networking using OkHttp 4, streaming JPEG frames to `POST /frame`.
- `app/src/main/AndroidManifest.xml`: Permissions declaration (`android.permission.CAMERA`, `android.permission.INTERNET`, `android.permission.ACCESS_NETWORK_STATE`, and cleartext HTTP traffic allowance for LAN IP addresses).

---

## Rebuilding the APK from Source

```bash
# On Linux/macOS:
./gradlew assembleDebug

# On Windows:
gradlew.bat assembleDebug
```
The output APK is generated at `app/build/outputs/apk/debug/app-debug.apk`.
