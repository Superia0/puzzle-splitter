# 拼图分割工具 (PuzzleSplitter)

把拼好的大图分割回若干原图的桌面工具。支持自动检测分割线、手动调整、网格均分，并能处理颜色/内容相似区域（如海天相接、街道拼接）。

## 功能特性

- **自动检测**：基于多特征融合（颜色差异 + 边缘投影 + 方差变化 + 直方图差异）自动识别拼图分割线
- **颜色相似区域友好**：直方图差异特征权重最高，即使整体颜色相近、纹理不同也能检出边界
- **网格均分**：按行列数快速均分图片
- **手动调整**：左键添加水平线、Shift+左键添加垂直线、右键删除最近线，也可通过列表精确管理
- **导出切分子图**：批量导出为 `split_01.png, split_02.png, ...`
- **保存标记图片**：导出带红/蓝分割线的预览图
- **简易中文 GUI**：基于 Tkinter，无需额外 UI 框架

## 环境依赖

- Python 3.11
- OpenCV (`opencv-python`)
- NumPy
- Pillow
- PyInstaller（仅打包需要）

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行程序

```bash
python main.py
```

### 3. 操作流程

1. 点击 **[打开图片]** 选择 jpg/png/bmp/webp 图片
2. 选择检测模式：
   - **自动检测**：拖动灵敏度滑块后点 **[执行检测]**
   - **网格均分**：输入行数、列数后点 **[执行检测]**
3. 在 Canvas 上手动调整：
   - **左键点击** → 添加水平分割线（y 坐标）
   - **Shift + 左键** → 添加垂直分割线（x 坐标）
   - **右键点击** → 删除距离点击位置最近的分割线
   - 也可在右侧列表中 **[添加]** / **[删除选中]** / **[清空]**
4. 点击 **[导出分割结果]** 选择目录，子图保存为 `split_01.png, split_02.png, ...`
5. （可选）点击 **[保存标记图片]** 保存带分割线的预览图

> 红线 = 水平分割线，蓝线 = 垂直分割线。所有列表坐标为原图坐标。

## 打包为 Windows 可执行文件（.exe）

### 前提条件
- Windows 10/11
- Python 3.11+（从 https://python.org 下载安装，**安装时勾选 "Add Python to PATH"**）

### 一键打包
1. 将 `main.py`、`splitter.py`、`requirements.txt`、`build.bat` 放在同一目录
2. **双击 `build.bat`**
3. 等待打包完成（首次约 2-3 分钟）
4. 在 `dist\` 目录中找到 `PuzzleSplitter.exe`

> `build.bat` 会自动安装依赖并调用 PyInstaller 打包为单文件 exe。

### 直接运行（无需打包）
如果已安装 Python，也可跳过打包直接运行：

```bat
pip install opencv-python numpy pillow
python main.py
```

## 算法原理

### 多特征融合

对图像的每一行/列计算 4 种特征曲线：

| 特征 | 含义 | 权重 |
|------|------|------|
| 颜色差异 | 行/列平均颜色欧氏距离（行平均降噪） | 0.35 |
| 边缘投影 | 高斯模糊+Canny边缘按行/列求和 | 0.10 |
| 方差变化率 | 滑动窗口(7px)方差的差分 | 0.10 |
| 直方图差异 | 块直方图(20~50px块)卡方距离 | 0.45 |

各特征归一化到 `[0,1]` 后加权融合，经自适应平滑(7~25px)后做峰值检测。

### 降噪设计（核心改进）

- **颜色差异**：先对每行取平均颜色（消除列方向随机噪点），再比较相邻行
- **直方图差异**：用块直方图（20~50px一块）代替逐行直方图，避免噪点导致的海量假阳性
- **边缘投影**：Canny前加高斯模糊，减少噪点产生的虚假边缘
- **突出度过滤**：峰值须比周围最小值高出阈值的一定比例，过滤渐变引起的低突出度波动

### 峰值检测

```
threshold = percentile(curve, 95 - sensitivity * 45)
min_distance = max(50, len(curve) // 20)
```

`sensitivity` 越高 → 阈值越低 → 检出更多候选线。融合曲线先经自适应平滑再找局部极大值，经突出度过滤和非极大值抑制后得到最终分割线。

## 文件结构

```
puzzle-splitter/
├── main.py          # 程序入口 + GUI（Tkinter）
├── splitter.py      # 分割算法核心（不依赖Tkinter，可独立测试）
├── requirements.txt # 依赖列表
├── build.bat        # Windows打包脚本
└── README.md        # 使用说明（本文件）
```

## 算法独立测试

`splitter.py` 可独立 import 和测试，无需 GUI：

```python
from splitter import SeamDetector
import numpy as np, cv2

img = cv2.imread("puzzle.png")
detector = SeamDetector(img)
h_lines = detector.detect_horizontal(sensitivity=0.3)
v_lines = detector.detect_vertical(sensitivity=0.3)
pieces = detector.split(h_lines, v_lines)
print(f"切分出 {len(pieces)} 块")
```

## 注意事项

- 大图片（>5000px）首次分析需要几秒钟计算特征，状态栏会显示进度提示
- 所有 GUI 文本为中文
- 导出图片使用 PNG 格式以避免压缩失真
- 分割线坐标基于原图，导出/切分使用原图分辨率
