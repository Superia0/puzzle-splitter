@echo off
chcp 65001 >nul 2>&1
echo ============================
echo  PuzzleSplitter Build Script
echo ============================
echo.
echo Installing dependencies...
pip install opencv-python numpy pillow pyinstaller
echo.
echo Building (first time takes 2-3 min)...
pyinstaller --onefile --windowed --name PuzzleSplitter --add-data "splitter.py;." --collect-all cv2 --hidden-import tkinter --hidden-import _tkinter main.py
echo.
if exist "dist\PuzzleSplitter.exe" (
    echo Done! Exe at: dist\PuzzleSplitter.exe
) else (
    echo Build failed, check errors above
)
echo.
pause
