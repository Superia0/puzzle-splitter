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

v2 改进：
- 峰锐度(FWHM)过滤：真实分割线是尖锐窄峰，内容边界是宽平隆起
- max/median比值绝对门槛：比值过低说明无显著分割信号
- 边缘保护区：忽略图像顶部/底部3%区域的假峰
- 极端宽高比更强抑制
- 最小峰间距防重复检测
"""

from __future__ import annotations

import numpy as np
import cv2


class SeamDetector:
    """拼图分割线检测器。

    通过多特征融合分析图像，自动检测水平/垂直分割线位置。
    """

    # 4种特征的融合权重（和为1.0）
    WEIGHT_COLOR = 0.35       # 颜色差异（行平均，已降噪）
    WEIGHT_EDGE = 0.10        # 边缘投影
    WEIGHT_VARIANCE = 0.10    # 方差变化率
    WEIGHT_HISTOGRAM = 0.45   # 直方图差异（块直方图，已降噪，权重最高）

    # 直方图bin数
    HIST_BINS = 32

    # 局部方差滑动窗口大小
    VARIANCE_WINDOW = 7

    # ===== v2 新增参数 =====

    # 峰锐度：FWHM占曲线最大长度的比例上限（超过则视为宽峰=内容边界）
    MAX_FWHM_RATIO = 0.04       # FWHM不能超过曲线长度的4%

    # 最大值/中位数比值的最小门槛（低于此值认为全曲线太平坦，无分割线）
    MIN_PEAK_TO_MEDIAN_RATIO = 3.5

    # 边缘保护区比例（曲线两端各忽略这么多）
    EDGE_MARGIN_RATIO = 0.05    # 忽略前5%和后5%

    # 相邻峰最小间距（占曲线长度比例，更近的只保留最强的一个）
    MIN_PEAK_GAP_RATIO = 0.03   # 至少相距3%

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
    # 特征计算（不变）
    # ------------------------------------------------------------------
    def _compute_color_diff(self):
        """颜色差异曲线：相邻行/列的平均颜色欧氏距离。

        先对每行/列取平均颜色（消除随机噪点），再比较相邻行/列。
        行平均天然消除列方向的随机噪点，只有系统性颜色突变（分割线）才会残留。
        """
        row_means = np.mean(self.float_image, axis=1)
        color_h = np.linalg.norm(np.diff(row_means, axis=0), axis=1)
        col_means = np.mean(self.float_image, axis=0)
        color_v = np.linalg.norm(np.diff(col_means, axis=0), axis=1)
        return color_h, color_v

    def _compute_edge_projection(self):
        """边缘投影曲线：Canny边缘按行/列求和。分割线处边缘密集。"""
        edge_h = np.sum(self.edges, axis=1).astype(np.float64)
        edge_v = np.sum(self.edges, axis=0).astype(np.float64)
        return edge_h, edge_v

    def _compute_variance_change(self):
        """局部方差变化率：滑动窗口方差的差分。"""
        var_h = self._local_variance_curve(self.gray.astype(np.float64), axis=0)
        var_v = self._local_variance_curve(self.gray.astype(np.float64), axis=1)
        change_h = np.abs(np.diff(var_h))
        change_v = np.abs(np.diff(var_v))
        return change_h, change_v

    def _local_variance_curve(self, gray_float: np.ndarray, axis: int):
        """对每一行（axis=0）或每一列（axis=1）计算局部方差曲线。"""
        win = self.VARIANCE_WINDOW
        half = win // 2

        if axis == 0:
            pad = np.pad(gray_float, ((half, half), (0, 0)), mode='reflect')
            P = pad.shape[0]
            cumsum = np.cumsum(pad, axis=0)
            cumsum_sq = np.cumsum(pad * pad, axis=0)

            win_len = P - win + 1
            win_sum = cumsum[win - 1:].copy()
            win_sum[1:] -= cumsum[:win_len - 1]
            win_sq_sum = cumsum_sq[win - 1:].copy()
            win_sq_sum[1:] -= cumsum_sq[:win_len - 1]

            mean = win_sum / win
            var = np.maximum(win_sq_sum / win - mean * mean, 0.0)
            var_curve = np.mean(var, axis=1)
            return var_curve
        else:
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
        """直方图差异曲线：相邻行/列灰度直方图卡方距离。"""
        hist_h = self._hist_diff_curve(self.gray, axis=0)
        hist_v = self._hist_diff_curve(self.gray, axis=1)
        return hist_h, hist_v

    def _hist_diff_curve(self, gray: np.ndarray, axis: int):
        """计算相邻块灰度直方图卡方距离曲线。"""
        bins = self.HIST_BINS
        H, W = gray.shape

        if axis == 0:
            length = H
            data = gray
        else:
            length = W
            data = gray.T

        block_size = max(20, min(50, length // 100))
        if block_size >= length:
            block_size = max(1, length // 3)
        n_blocks = length // block_size
        if n_blocks < 2:
            return np.zeros(max(0, length - 1), dtype=np.float64)

        hists = np.zeros((n_blocks, bins), dtype=np.float32)
        for i in range(n_blocks):
            block = data[i * block_size:(i + 1) * block_size]
            h = cv2.calcHist([block], [0], None, [bins], [0, 256])
            hists[i] = h.flatten()
        diffs = np.array([
            cv2.compareHist(hists[i].astype(np.float32),
                            hists[i + 1].astype(np.float32),
                            cv2.HISTCMP_CHISQR)
            for i in range(n_blocks - 1)
        ], dtype=np.float64)
        x_old = np.arange(len(diffs)) * block_size + block_size
        x_new = np.arange(length - 1)
        if len(diffs) >= 2:
            return np.interp(x_new, x_old, diffs)
        else:
            return np.full(length - 1, diffs[0] if len(diffs) >= 1 else 0.0)

    # ------------------------------------------------------------------
    # 融合与峰值检测（v2 重写）
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
        """4种特征归一化后加权融合。"""
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
        # 自适应平滑
        smooth_w = max(9, min(25, len(fused) // 200))
        if len(fused) >= smooth_w:
            kernel = np.ones(smooth_w) / smooth_w
            fused = np.convolve(fused, kernel, mode='same')
        return fused

    @staticmethod
    def _calc_fwhm(curve: np.ndarray, peak_pos: int) -> float:
        """计算峰值在半高处的全宽度(FWHM)。

        从峰值位置向左右搜索，找到曲线值降到峰值一半的位置。
        返回宽度（像素数）。如果搜索到边界则返回一个大值。
        """
        peak_val = curve[peak_pos]
        half_val = peak_val / 2.0
        n = len(curve)

        # 向左搜索
        left = peak_pos
        while left > 0 and curve[left] > half_val:
            left -= 1

        # 向右搜索
        right = peak_pos
        while right < n - 1 and curve[right] > half_val:
            right += 1

        return float(right - left)

    @staticmethod
    def _find_peaks(curve: np.ndarray, sensitivity: float,
                    edge_margin_ratio: float = 0.03,
                    max_fwhm_ratio: float = 0.04,
                    min_peak_ratio: float = 3.5,
                    min_gap_ratio: float = 0.03) -> list:
        """v2 峰值检测：多维度过滤确保只检出真实分割线。

        过滤流程（按顺序，逐步收紧）：
        0. 前置检查：曲线太短直接返回空
        1. 全局显著性检查：max/median 比值不足 → 无分割信号 → 返回空
        2. 候选检测：百分位阈值 + 局部极大值 + 突出度
        3. 边缘保护区过滤：剔除靠近曲线两端的候选（图片边框伪影）
        4. 中位数倍数阈值：剔除接近中位数的噪声峰
        5. 峰锐度(FWHM)过滤：剔除过宽的峰（内容边界产生的宽隆起）
        6. 最小间距合并：过近的峰只保留最强的

        Args:
            curve: 融合特征曲线（已归一化到[0,1]附近）
            sensitivity: 灵敏度 0~1，越大越宽松
            edge_margin_ratio: 边缘保护区比例
            max_fwhm_ratio: FWHM占曲线长度的最大允许比例
            min_peak_ratio: max/median 最小比值门槛
            min_gap_ratio: 相邻峰最小间距比例
        """
        n = len(curve)
        if n < 20:
            return []

        # ===== Step 0 & 1: 全局显著性检查 =====
        median_val = float(np.median(curve))
        max_val = float(curve.max())

        if median_val < 1e-10:
            # 曲线几乎全零，没有任何信号
            return []

        peak_to_median = max_val / median_val
        if peak_to_median < min_peak_ratio:
            # 整个曲线太平坦，max 和 median 差距不够大
            # 说明没有显著的分割线信号（所有波动都是同级别的内容噪声）
            return []

        # ===== Step 2: 候选峰值检测 =====
        threshold = float(np.percentile(curve, 95 - sensitivity * 45))
        min_distance = max(30, n // 25)

        candidates = []
        for i in range(1, n - 1):
            if curve[i] > threshold and curve[i] >= curve[i - 1] and curve[i] > curve[i + 1]:
                left = curve[max(0, i - min_distance):i]
                right = curve[i + 1:i + 1 + min_distance]
                base = max(left.min() if len(left) > 0 else 0.0,
                           right.min() if len(right) > 0 else 0.0)
                prominence = curve[i] - base
                if prominence < threshold * 0.15:
                    continue
                candidates.append((i, curve[i], prominence))

        if not candidates:
            return []

        # NMS
        candidates.sort(key=lambda x: x[0])
        suppressed = []
        for pos, val, prom in candidates:
            if suppressed and pos - suppressed[-1][0] < min_distance:
                if val > suppressed[-1][1]:
                    suppressed[-1] = (pos, val, prom)
            else:
                suppressed.append((pos, val, prom))

        # ===== Step 3: 边缘保护区过滤 =====
        edge_margin = int(n * edge_margin_ratio)
        suppressed = [
            p for p in suppressed
            if edge_margin <= p[0] < n - edge_margin
        ]
        if not suppressed:
            return []

        # ===== Step 4: 中位数倍数阈值 =====
        # sensitivity=0.05 → multiplier≈1.93（严格）
        # sensitivity=0.5  → multiplier=1.25
        # sensitivity=1.0  → multiplier=0.5（最宽松）
        multiplier = 2.0 - sensitivity * 1.5
        median_threshold = median_val * (1.0 + multiplier)

        filtered = [p for p in suppressed if p[1] >= median_threshold]
        if not filtered:
            return []

        # ===== Step 4.5: 相对强度过滤 =====
        # 真实分割线的信号应该显著强于其他候选峰。
        # 如果最强峰值为 M，则每个峰至少需要达到 M × rel_min_ratio，
        # 否则视为内容边界的弱信号而剔除。
        # rel_min_ratio 随灵敏度变化：低灵敏度更严格（只保留最显著的峰）。
        #   sens=0.05 → rel_min=0.48 （严格：只保留>48%最强峰的）
        #   sens=0.50 → rel_min=0.33 （中等）
        #   sens=1.00 → rel_min=0.15 （宽松）
        rel_min_ratio = 0.50 - sensitivity * 0.35
        max_filtered_val = max(p[1] for p in filtered)
        abs_threshold = max_filtered_val * rel_min_ratio
        rel_filtered = [p for p in filtered if p[1] >= abs_threshold]
        if not rel_filtered:
            # 相对过滤后全被淘汰，回退到仅用中位数阈值的结果
            rel_filtered = filtered

        # ===== Step 5: 峰锐度(FWHM)过滤 =====
        fwhm_limit = int(n * max_fwhm_ratio)
        sharp_peaks = []
        for pos, val, prom in rel_filtered:
            fwhm = SeamDetector._calc_fwhm(curve, pos)
            if fwhm <= fwhm_limit:
                sharp_peaks.append((pos, val, prom, fwhm))

        if not sharp_peaks:
            # 所有峰都太宽了，不是真正的分割线
            return []

        # ===== Step 6: 最小间距合并 =====
        min_gap = int(n * min_gap_ratio)
        sharp_peaks.sort(key=lambda x: x[1], reverse=True)  # 按强度排序
        final_peaks = []
        used = set()
        for pos, val, prom, fwhm in sharp_peaks:
            if any(abs(pos - u) < min_gap for u in used):
                continue
            final_peaks.append(pos)
            used.add(pos)

        final_peaks.sort()
        return final_peaks

    # ------------------------------------------------------------------
    # 公开API
    # ------------------------------------------------------------------
    def detect_horizontal(self, sensitivity: float = 0.5, max_peaks: int = 0) -> list:
        """检测水平分割线，返回y坐标列表。

        v2 增强：
        - 宽高比启发式更激进：宽图(w>h×2)大幅降低水平灵敏度
        - 极端比例(w>h×3)几乎禁止水平检测
        """
        aspect = self.w / self.h  # 越宽→越不需要水平分割线
        if aspect > 3.0:
            eff_sens = sensitivity * 0.1      # 几乎禁止
        elif aspect > 2.0:
            eff_sens = sensitivity * 0.25     # 大幅降低
        elif aspect > 1.5:
            eff_sens = sensitivity * 0.55     # 适度降低
        else:
            eff_sens = sensitivity

        peaks = self._find_peaks(self._fused_h, eff_sens)
        if max_peaks > 0 and len(peaks) > max_peaks:
            ranked = sorted(peaks, key=lambda p: self._fused_h[p], reverse=True)
            peaks = ranked[:max_peaks]
        return sorted(p + 1 for p in peaks)

    def detect_vertical(self, sensitivity: float = 0.5, max_peaks: int = 0) -> list:
        """检测垂直分割线，返回x坐标列表。

        v2 增强：
        - 高宽比启发式更激进：高图(h>w×2)大幅降低垂直灵敏度
        - 极端比例(h>w×3)几乎禁止垂直检测
        """
        aspect = self.h / self.w  # 越高→越不需要垂直分割线
        if aspect > 3.0:
            eff_sens = sensitivity * 0.1      # 几乎禁止
        elif aspect > 2.0:
            eff_sens = sensitivity * 0.25     # 大幅降低
        elif aspect > 1.5:
            eff_sens = sensitivity * 0.55     # 适度降低
        else:
            eff_sens = sensitivity

        peaks = self._find_peaks(self._fused_v, eff_sens)
        if max_peaks > 0 and len(peaks) > max_peaks:
            ranked = sorted(peaks, key=lambda p: self._fused_v[p], reverse=True)
            peaks = ranked[:max_peaks]
        return sorted(p + 1 for p in peaks)

    def get_curves(self) -> dict:
        """返回所有特征曲线数据，用于可视化调试。"""
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
