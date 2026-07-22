"""拼图分割算法核心模块 - v3 (行业方法：块对比 + 自适应阈值)。

v3 改进：
1. 块统计对比（200行大窗口）：捕捉"内容差异"而非"像素噪声"
2. 自适应百分位阈值：避免强峰拉高整体阈值
3. 方向抑制：极端宽高比完全抑制另一方向
4. 预期峰数：根据宽高比自动估计，优先保留最强峰

对于无缝拼接图（如婚礼照片竖排），自动检测可能产生假阳性，
建议使用 Grid Split 模式（已知网格布局时更准确）。
"""

from __future__ import annotations

import numpy as np
import cv2


class SeamDetector:
    """拼图分割线检测器 (v3)。"""

    BLOCK_SIZE = 200
    CANNY_THRESH1 = 50
    CANNY_THRESH2 = 150

    def __init__(self, image: np.ndarray):
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image必须是BGR格式的3通道ndarray")

        self.image = image
        self.float_image = image.astype(np.float64)
        self.h, self.w = image.shape[:2]
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.edges = cv2.Canny(
            cv2.GaussianBlur(self.gray, (5, 5), 0),
            self.CANNY_THRESH1, self.CANNY_THRESH2
        )

        self._mean_diff_h, self._std_diff_h, self._edge_diff_h = self._compute_block_stats(0)
        self._mean_diff_v, self._std_diff_v, self._edge_diff_v = self._compute_block_stats(1)
        self._color_diff_h, self._color_diff_v = self._compute_color_diff()

    def _compute_block_stats(self, axis: int):
        """用 cumsum 高效计算块均值差、标准差差、边缘密度差。"""
        B = self.BLOCK_SIZE
        if axis == 0:
            length, data = self.h, self.float_image
            edge_data = self.edges.astype(np.float64)
        else:
            length, data = self.w, self.float_image.transpose(1, 0, 2)
            edge_data = self.edges.astype(np.float64).T

        cumsum = np.cumsum(data, axis=0)
        cumsum_sq = np.cumsum(data * data, axis=0)
        edge_cumsum = np.cumsum(edge_data, axis=0)
        count = B * data.shape[1]

        mean_diff = np.zeros(length)
        std_diff = np.zeros(length)
        edge_diff = np.zeros(length)

        for y in range(B, length - B):
            if y == B:
                sa, sqa, ea = cumsum[B-1], cumsum_sq[B-1], edge_cumsum[B-1]
            else:
                sa = cumsum[y-1] - cumsum[y-B-1]
                sqa = cumsum_sq[y-1] - cumsum_sq[y-B-1]
                ea = edge_cumsum[y-1] - edge_cumsum[y-B-1]
            sb = cumsum[y+B-1] - cumsum[y-1]
            sqb = cumsum_sq[y+B-1] - cumsum_sq[y-1]
            eb = edge_cumsum[y+B-1] - edge_cumsum[y-1]

            ma, mb = sa/count, sb/count
            va = np.maximum(sqa/count - ma*ma, 0.0)
            vb = np.maximum(sqb/count - mb*mb, 0.0)
            mean_diff[y] = np.mean(np.abs(ma - mb))
            std_diff[y] = np.mean(np.abs(np.sqrt(va) - np.sqrt(vb)))
            edge_diff[y] = np.abs(ea - eb).sum() / count

        return mean_diff, std_diff, edge_diff

    def _compute_color_diff(self):
        row_means = np.mean(self.float_image, axis=1)
        color_h = np.linalg.norm(np.diff(row_means, axis=0), axis=1)
        col_means = np.mean(self.float_image, axis=0)
        color_v = np.linalg.norm(np.diff(col_means, axis=0), axis=1)
        return color_h, color_v

    @staticmethod
    def _normalize(arr):
        mn, mx = float(arr.min()), float(arr.max())
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    def _fuse(self, mean_d, std_d, edge_d, color_d):
        """融合4种块统计特征。std_diff 权重最高（纹理变化最能反映内容切换）。"""
        ml = min(len(mean_d), len(std_d), len(edge_d), len(color_d))
        return (0.2 * self._normalize(mean_d[:ml]) +
                0.5 * self._normalize(std_d[:ml]) +
                0.2 * self._normalize(edge_d[:ml]) +
                0.1 * self._normalize(color_d[:ml]))

    def _find_peaks(self, curve, sensitivity, expected_count=0):
        """百分位阈值 + 局部极大值 + NMS + 边缘保护。"""
        n = len(curve)
        if n < 20:
            return []

        # 轻平滑
        sw = 5
        if n >= sw:
            curve = np.convolve(curve, np.ones(sw)/sw, mode='same')

        max_val = float(curve.max())
        min_dist = max(50, n // 20)
        margin = max(1, int(n * 0.05))

        # 百分位阈值
        p = int(85 - sensitivity * 60)
        threshold = float(np.percentile(curve, p))

        candidates = []
        for i in range(margin, n - margin):
            if curve[i] > threshold and curve[i] >= curve[i-1] and curve[i] > curve[i+1]:
                left = curve[max(0, i-min_dist):i]
                right = curve[i+1:i+1+min_dist]
                base = max(left.min() if len(left) else 0, right.min() if len(right) else 0)
                if curve[i] - base > 0.03 * max_val:
                    candidates.append((i, curve[i]))

        if not candidates:
            return []

        # NMS
        candidates.sort(key=lambda x: x[0])
        suppressed = []
        for pos, val in candidates:
            if suppressed and pos - suppressed[-1][0] < min_dist:
                if val > suppressed[-1][1]:
                    suppressed[-1] = (pos, val)
            else:
                suppressed.append((pos, val))

        # 按强度排序
        suppressed.sort(key=lambda x: x[1], reverse=True)

        # 限制峰数
        if expected_count > 0:
            suppressed = suppressed[:expected_count]

        final = sorted(p[0] for p in suppressed)
        return final

    def detect_horizontal(self, sensitivity=0.5, max_peaks=0):
        aspect = self.w / self.h
        if aspect > 3.0:
            return []
        elif aspect > 2.0:
            eff_sens = sensitivity * 0.25
        elif aspect > 1.5:
            eff_sens = sensitivity * 0.55
        else:
            eff_sens = sensitivity

        fused = self._fuse(self._mean_diff_h, self._std_diff_h, self._edge_diff_h, self._color_diff_h)

        # 预期峰数：高图预期 int(h/w) 条水平线
        expected = max(1, int(self.h / self.w)) if self.h > self.w * 1.5 else 0
        if max_peaks > 0:
            expected = max_peaks

        peaks = self._find_peaks(fused, eff_sens, expected)
        if max_peaks > 0 and len(peaks) > max_peaks:
            peaks = sorted(peaks, key=lambda p: fused[p], reverse=True)[:max_peaks]
        return sorted(p + 1 for p in peaks)

    def detect_vertical(self, sensitivity=0.5, max_peaks=0):
        aspect = self.h / self.w
        if aspect > 3.0:
            return []
        elif aspect > 2.0:
            eff_sens = sensitivity * 0.25
        elif aspect > 1.5:
            eff_sens = sensitivity * 0.55
        else:
            eff_sens = sensitivity

        fused = self._fuse(self._mean_diff_v, self._std_diff_v, self._edge_diff_v, self._color_diff_v)

        expected = max(1, int(self.w / self.h)) if self.w > self.h * 1.5 else 0
        if max_peaks > 0:
            expected = max_peaks

        peaks = self._find_peaks(fused, eff_sens, expected)
        if max_peaks > 0 and len(peaks) > max_peaks:
            peaks = sorted(peaks, key=lambda p: fused[p], reverse=True)[:max_peaks]
        return sorted(p + 1 for p in peaks)

    def get_curves(self):
        return {
            'mean_diff_h': self._mean_diff_h, 'mean_diff_v': self._mean_diff_v,
            'std_diff_h': self._std_diff_h, 'std_diff_v': self._std_diff_v,
            'edge_diff_h': self._edge_diff_h, 'edge_diff_v': self._edge_diff_v,
            'color_h': self._color_diff_h, 'color_v': self._color_diff_v,
        }

    def split(self, h_lines, v_lines):
        h_bounds = [0] + sorted(h_lines) + [self.h]
        v_bounds = [0] + sorted(v_lines) + [self.w]
        pieces = []
        for i in range(len(h_bounds) - 1):
            for j in range(len(v_bounds) - 1):
                pieces.append(self.image[h_bounds[i]:h_bounds[i+1], v_bounds[j]:v_bounds[j+1]])
        return pieces
