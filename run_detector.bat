@echo off
setlocal
title AI Road Hazard & Campus Waste Detector
color 0A

:: Ensure Python is in PATH
set "PATH=E:\;E:\Scripts;%PATH%"

echo =================================================================
echo   AI ROAD HAZARD & CAMPUS WASTE DETECTOR (Potholes & Trash)
echo =================================================================
echo.
echo Select Video Input Source:
echo   [1] Laptop Webcam
echo   [2] ESP32-CAM Wi-Fi Stream
echo   [3] Auto-Detect (Webcam or ESP32)
echo   [4] Custom Video File
echo.
set "choice=3"
set /p choice="Enter choice 1, 2, 3, or 4 [Default: 3]: "

if "%choice%"=="1" goto USE_WEBCAM
if "%choice%"=="2" goto USE_ESP32
if "%choice%"=="4" goto USE_VIDEO
goto AUTO_DETECT

:USE_WEBCAM
echo.
echo Launching with local Webcam [source 0]...
py -u road_hazard_detector.py --source 0
goto DONE

:USE_ESP32
echo.
echo Launching with ESP32-CAM stream [192.168.4.1]...
py -u road_hazard_detector.py --source http://192.168.4.1:81/stream
goto DONE

:USE_VIDEO
echo.
set /p vpath="Enter video file path: "
py -u road_hazard_detector.py --source "%vpath%"
goto DONE

:AUTO_DETECT
echo.
echo Launching with Smart Auto-Detection...
py -u road_hazard_detector.py
goto DONE

:DONE
echo.
echo =================================================================
echo Detection finished. Window will stay open for review.
echo =================================================================
pause
