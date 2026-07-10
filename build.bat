@echo off
chcp 65001
echo ============================
echo  拼图分割工具 - 打包脚本
echo ============================
echo.
echo 正在安装依赖...
pip install opencv-python numpy pillow pyinstaller
echo.
echo 正在打包（首次约2-3分钟，请耐心等待）...
pyinstaller --onefile --windowed --name PuzzleSplitter --add-data "splitter.py;." --collect-all cv2 --hidden-import tkinter --hidden-import _tkinter main.py
echo.
if exist "dist\PuzzleSplitter.exe" (
    echo 打包成功！可执行文件位于: dist\PuzzleSplitter.exe
) else (
    echo 打包可能失败，请检查上方错误信息
)
echo.
pause
