"""拼图分割算法核心模块。

本模块不依赖任何GUI库（如Tkinter），可独立import和测试。
仅依赖numpy、opencv-python。

核心思路：对图像的每一行/列计算4种特征曲线，归一化后加权融合，
在融合曲线上做峰值检测得到分割线位置。

4种特征：
1. 颜色差异曲线 - 相邻行/列的平均RGB欧氏距离
2. 边缘投影曲线 - Canny边缘按行/列求和
3. 局部方差变化率 - 滑动窗口方差的差分
4. 直方图差异曲线 - 相邻行/列灰度直方图卡方距离（处理颜色相似区域的关键）
"""

from __future__ import annotations

import numpy as np
import cv2


class SeamDetector:
    """拼图分割线检测器。

    通过多特征融合分析图像，自动检测水平/垂直分割线位置。
    """

    # 4种特征的融合权重（和为1.0）
    # 颜色和直方图已做行平均/块化降噪，权重更高
    WEIGHT_COLOR = 0.35       # 颜色差异（行平均，已降噪）
    WEIGHT_EDGE = 0.10        # 边缘投影
    WEIGHT_VARIANCE = 0.10    # 方差变化率
    WEIGHT_HISTOGRAM = 0.45   # 直方图差异（块直方图，已降噪，权重最高）

    # 直方图bin数
    HIST_BINS = 32

    # 局部方差滑动窗口大小
    VARIANCE_WINDOW = 7

    def __init__(self, image: np.ndarray):
        """初始化并预计算所有特征曲线。

        Args:
            image: BGR格式ndarray，来自cv2.imread。形状(H, W, 3)。
        """
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image必须是BGR格式的3通道ndarray")

        # 转为float64避免计算时溢出
        self.image = image
        self.float_image = image.astype(np.float64)
        self.h, self.w = image.shape[:2]

        # 灰度图
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Canny边缘（先高斯模糊降噪，避免噪点产生过多边缘）
        self.edges = cv2.Canny(cv2.GaussianBlur(self.gray, (5, 5), 0), 50, 150)

        # 预计算4种特征曲线（水平方向=按行分析；垂直方向=按列分析）
        self._color_h, self._color_v = self._compute_color_diff()
        self._edge_h, self._edge_v = self._compute_edge_projection()
        self._var_h, self._var_v = self._compute_variance_change()
        self._hist_h, self._hist_v = self._compute_histogram_diff()

        # 融合曲线
        self._fused_h = self._fuse(
            self._color_h, self._edge_h, self._var_h, self._hist_h
        )
        self._fused_v = self._fuse(
            self._color_v, self._edge_v, self._var_v, self._hist_v
        )

    # ------------------------------------------------------------------
    # 特征计算
    # ------------------------------------------------------------------
    def _compute_color_diff(self):
        """颜色差异曲线：相邻行/列的平均颜色欧氏距离。

        先对每行/列取平均颜色（消除随机噪点），再比较相邻行/列。
        行平均天然消除列方向的随机噪点，只有系统性颜色突变（分割线）才会残留。
        """
        # 行方向：每行取平均BGR → (H, 3)，相邻行平均颜色欧氏距离 → (H-1,)
        row_means = np.mean(self.float_image, axis=1)
        color_h = np.linalg.norm(np.diff(row_means, axis=0), axis=1)
        # 列方向：每列取平均BGR → (W, 3)，相邻列平均颜色欧氏距离 → (W-1,)
        col_means = np.mean(self.float_image, axis=0)
        color_v = np.linalg.norm(np.diff(col_means, axis=0), axis=1)
        return color_h, color_v

    def _compute_edge_projection(self):
        """边缘投影曲线：Canny边缘按行/列求和。分割线处边缘密集。"""
        edge_h = np.sum(self.edges, axis=1).astype(np.float64)  # (H,)
        edge_v = np.sum(self.edges, axis=0).astype(np.float64)  # (W,)
        return edge_h, edge_v

    def _compute_variance_change(self):
        """局部方差变化率：滑动窗口方差的差分。

        对每行/列计算局部窗口方差，返回方差的绝对差分（不是方差本身）。
        """
        var_h = self._local_variance_curve(self.gray.astype(np.float64), axis=0)
        var_v = self._local_variance_curve(self.gray.astype(np.float64), axis=1)
        # 返回方差的变化率（差分绝对值）
        change_h = np.abs(np.diff(var_h))
        change_v = np.abs(np.diff(var_v))
        return change_h, change_v

    def _local_variance_curve(self, gray_float: np.ndarray, axis: int):
        """对每一行（axis=0）或每一列（axis=1）计算局部方差曲线。

        使用reflect填充+累加和加速滑动窗口方差计算。
        返回长度等于该方向尺寸的方差曲线。
        """
        win = self.VARIANCE_WINDOW
        half = win // 2

        if axis == 0:
            # 每一行的方差：在垂直方向上取滑动窗口，再对列求均值
            pad = np.pad(gray_float, ((half, half), (0, 0)), mode='reflect')
            P = pad.shape[0]  # P = H + 2*half
            cumsum = np.cumsum(pad, axis=0)
            cumsum_sq = np.cumsum(pad * pad, axis=0)

            # 窗口和: win_sum[i] = cumsum[i+win-1] - cumsum[i-1] (i>=1)
            # 长度 = P - win + 1 = H
            win_len = P - win + 1
            win_sum = cumsum[win - 1:].copy()              # win_sum[i]=cumsum[i+win-1]
            win_sum[1:] -= cumsum[:win_len - 1]            # 减去 cumsum[i-1]
            win_sq_sum = cumsum_sq[win - 1:].copy()
            win_sq_sum[1:] -= cumsum_sq[:win_len - 1]

            mean = win_sum / win
            var = np.maximum(win_sq_sum / win - mean * mean, 0.0)
            var_curve = np.mean(var, axis=1)
            return var_curve
        else:
            # 每一列的方差：在水平方向上取滑动窗口，再对行求均值
            pad = np.pad(gray_float, ((0, 0), (half, half)), mode='reflect')
            P = pad.shape[1]
            cumsum = np.cumsum(pad, axis=1)
            cumsum_sq = np.cumsum(pad * pad, axis=1)

            win_len = P - win + 1
            win_sum = cumsum[:, win - 1:].copy()
            win_sum[:, 1:] -= cumsum[:, :win_len - 1]
            win_sq_sum = cumsum_sq[:, win - 1:].copy()
            win_sq_sum[:, 1:] -= cumsum_sq[:, :win_len - 1]

            mean = win_sum / win
            var = np.maximum(win_sq_sum / win - mean * mean, 0.0)
            var_curve = np.mean(var, axis=0)
            return var_curve

    def _compute_histogram_diff(self):
        """直方图差异曲线：相邻行/列灰度直方图卡方距离。

        处理颜色相似区域的关键：即使整体颜色相似，纹理分布不同时
        直方图差异仍然显著。
        """
        hist_h = self._hist_diff_curve(self.gray, axis=0)
        hist_v = self._hist_diff_curve(self.gray, axis=1)
        return hist_h, hist_v

    def _hist_diff_curve(self, gray: np.ndarray, axis: int):
        """计算相邻块灰度直方图卡方距离曲线。

        使用块直方图代替逐行直方图，消除噪点引起的假阳性。
        块大小自适应：图像越大块越大（20~50px），插值回原长度。
        """
        bins = self.HIST_BINS
        H, W = gray.shape

        if axis == 0:
            length = H
            data = gray
        else:
            length = W
            data = gray.T  # 转置后按行处理

        block_size = max(20, min(50, length // 100))
        if block_size >= length:
            block_size = max(1, length // 3)
        n_blocks = length // block_size
        if n_blocks < 2:
            return np.zeros(max(0, length - 1), dtype=np.float64)

        # 计算每个块的直方图
        hists = np.zeros((n_blocks, bins), dtype=np.float32)
        for i in range(n_blocks):
            block = data[i * block_size:(i + 1) * block_size]
            h = cv2.calcHist([block], [0], None, [bins], [0, 256])
            hists[i] = h.flatten()
        # 相邻块卡方距离
        diffs = np.array([
            cv2.compareHist(hists[i].astype(np.float32),
                            hists[i + 1].astype(np.float32),
                            cv2.HISTCMP_CHISQR)
            for i in range(n_blocks - 1)
        ], dtype=np.float64)
        # 插值到 length-1
        x_old = np.arange(len(diffs)) * block_size + block_size
        x_new = np.arange(length - 1)
        if len(diffs) >= 2:
            return np.interp(x_new, x_old, diffs)
        else:
            return np.full(length - 1, diffs[0] if len(diffs) >= 1 else 0.0)

    # ------------------------------------------------------------------
    # 融合与峰值检测
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        """将数组归一化到[0,1]。若max==min则返回全0。"""
        amin = float(arr.min())
        amax = float(arr.max())
        if amax - amin < 1e-12:
            return np.zeros_like(arr)
        return (arr - amin) / (amax - amin)

    def _fuse(self, color, edge, var, hist):
        """4种特征归一化后加权融合。

        注意各曲线长度可能不同（差分会让长度少1），统一取最短长度对齐。
        """
        # 对齐到最短长度
        min_len = min(len(color), len(edge), len(var), len(hist))
        color = self._normalize(color[:min_len])
        edge = self._normalize(edge[:min_len])
        var = self._normalize(var[:min_len])
        hist = self._normalize(hist[:min_len])

        fused = (
            self.WEIGHT_COLOR * color
            + self.WEIGHT_EDGE * edge
            + self.WEIGHT_VARIANCE * var
            + self.WEIGHT_HISTOGRAM * hist
        )
        # 自适应平滑：曲线越长平滑窗口越大，消除噪点波动但保留分割线峰值
        smooth_w = max(7, min(25, len(fused) // 300))
        if len(fused) >= smooth_w:
            kernel = np.ones(smooth_w) / smooth_w
            fused = np.convolve(fused, kernel, mode='same')
        return fused

    @staticmethod
    def _find_peaks(curve: np.ndarray, sensitivity: float):
        """峰值检测：在融合曲线上找局部极大值。

        Args:
            curve: 融合后的1D曲线
            sensitivity: 0.0-1.0，越高越敏感（阈值越低，检出更多峰值）

        Returns:
            峰值位置列表（升序）
        """
        if len(curve) < 3:
            return []

        # sensitivity越高→越敏感→阈值越低→检出更多
        threshold = float(np.percentile(curve, 95 - sensitivity * 45))
        min_distance = max(50, len(curve) // 20)

        peaks = []
        n = len(curve)
        for i in range(1, n - 1):
            if curve[i] > threshold and curve[i] >= curve[i - 1] and curve[i] > curve[i + 1]:
                # 突出度过滤：峰值须比周围min_distance范围内的最小值高出一定比例
                # 过滤渐变/噪点引起的低突出度波动
                left = curve[max(0, i - min_distance):i]
                right = curve[i + 1:i + 1 + min_distance]
                base = max(left.min() if len(left) > 0 else 0.0,
                           right.min() if len(right) > 0 else 0.0)
                prominence = curve[i] - base
                if prominence < threshold * 0.15:
                    continue
                # 非极大值抑制：距上一个峰值不足min_distance则取较大者
                if peaks and i - peaks[-1] < min_distance:
                    if curve[i] > curve[peaks[-1]]:
                        peaks[-1] = i
                else:
                    peaks.append(i)
        return peaks

    # ------------------------------------------------------------------
    # 公开API
    # ------------------------------------------------------------------
    def detect_horizontal(self, sensitivity: float = 0.5) -> list:
        """检测水平分割线，返回y坐标列表。

        Args:
            sensitivity: 0.0-1.0，越高越敏感（检出更多线）。

        Returns:
            y坐标列表（升序）。
        """
        # 融合曲线由diff得到，峰值索引i表示行i与行i+1之间的差异
        # 分割线实际位置在 i+1（行i+1是新区域的起始）
        peaks = self._find_peaks(self._fused_h, sensitivity)
        return sorted(p + 1 for p in peaks)

    def detect_vertical(self, sensitivity: float = 0.5) -> list:
        """检测垂直分割线，返回x坐标列表。

        Args:
            sensitivity: 0.0-1.0，越高越敏感（检出更多线）。

        Returns:
            x坐标列表（升序）。
        """
        # 融合曲线由diff得到，峰值索引i表示列i与列i+1之间的差异
        # 分割线实际位置在 i+1
        peaks = self._find_peaks(self._fused_v, sensitivity)
        return sorted(p + 1 for p in peaks)

    def get_curves(self) -> dict:
        """返回所有特征曲线数据，用于可视化调试。

        Returns:
            dict，包含color/edge/var/hist/fused各方向曲线。
        """
        return {
            'color_h': self._color_h,
            'color_v': self._color_v,
            'edge_h': self._edge_h,
            'edge_v': self._edge_v,
            'var_h': self._var_h,
            'var_v': self._var_v,
            'hist_h': self._hist_h,
            'hist_v': self._hist_v,
            'fused_h': self._fused_h,
            'fused_v': self._fused_v,
        }

    def split(self, h_lines: list, v_lines: list) -> list:
        """按分割线切分图片，返回子图列表（按行优先顺序）。

        Args:
            h_lines: 水平分割线y坐标列表
            v_lines: 垂直分割线x坐标列表

        Returns:
            子图ndarray列表，按行优先顺序（第一行从左到右，再第二行...）
        """
        h_bounds = [0] + sorted(h_lines) + [self.h]
        v_bounds = [0] + sorted(v_lines) + [self.w]
        pieces = []
        for i in range(len(h_bounds) - 1):
            for j in range(len(v_bounds) - 1):
                piece = self.image[
                    h_bounds[i]:h_bounds[i + 1],
                    v_bounds[j]:v_bounds[j + 1],
                ]
                pieces.append(piece)
        return pieces
