@echo off
set "PYTHON_PATH=C:\Users\ASUS\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.8_qbz5n2kfra8p0\python.exe"
if exist "%PYTHON_PATH%" (
    echo Starting Face Emotion Detection using Python 3.8...
    "%PYTHON_PATH%" face_emotion_detection.py
) else (
    echo Error: Python 3.8 not found at the expected location.
    echo Please ensure Python 3.8 is installed from the Microsoft Store.
    pause
)
