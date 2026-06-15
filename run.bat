@echo off
echo Multimodal Emergency Vehicle Detection System
echo =============================================
echo.
echo Choose an option:
echo 1. Run multimodal system on webcam
echo 2. Run multimodal system on video
echo 3. Run YOLO detection only
echo 4. Evaluate models
echo 5. Train YOLO (if needed)
echo 6. Train audio (if needed)
echo.
set /p choice="Enter choice (1-6): "

if "%choice%"=="1" (
    python scripts\multimodal_system.py --webcam
) else if "%choice%"=="2" (
    set /p video="Enter video path: "
    python scripts\multimodal_system.py --video %video%
) else if "%choice%"=="3" (
    set /p video="Enter video path: "
    python scripts\video_detection.py --source %video%
) else if "%choice%"=="4" (
    python scripts\evaluate.py
) else if "%choice%"=="5" (
    python scripts\train_yolo.py
) else if "%choice%"=="6" (
    python scripts\train_audio.py
) else (
    echo Invalid choice
)

pause