"""拼图分割工具 - GUI主程序 (Apple风格UI)。

基于Tkinter，采用Apple原生应用设计语言：
- 浅灰白窗口背景 #F5F5F7
- 白色圆角卡片分组
- Apple Blue #007AFF 主色调
- MiSans / Microsoft YaHei UI 字体
- 浅色画布背景 #F0F0F2
"""

from __future__ import annotations

import os
import threading
import tkinter.font as tkfont

import numpy as np
import cv2
from PIL import Image, ImageTk

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, simpledialog
except ImportError as e:
    raise SystemExit("需要Tkinter支持。Windows版Python通常自带，Linux请安装 python3-tk。") from e

from splitter import SeamDetector


# ============================================================
# Apple风格配色常量
# ============================================================
C_WINDOW_BG = '#F5F5F7'       # 窗口背景
C_CARD_BG = '#FFFFFF'          # 卡片背景
C_CARD_BORDER = '#E5E5EA'      # 卡片边框
C_CANVAS_BG = '#F0F0F2'        # 画布背景
C_BLUE = '#007AFF'             # Apple Blue 主色
C_BLUE_DARK = '#0051D5'        # 按钮按下
C_RED = '#FF3B30'              # 水平分割线
C_TEXT = '#1D1D1F'             # 主文字
C_TEXT_SEC = '#86868B'         # 次要文字
C_BTN_SEC_BG = '#E5E5EA'       # 次要按钮
C_BTN_SEC_DARK = '#D1D1D6'     # 次要按钮按下
C_LIST_BG = '#F5F5F7'          # 列表背景
C_STATUS_BG = '#FAFAFA'        # 状态栏背景
C_STATUS_BORDER = '#E5E5EA'    # 状态栏边框


def _detect_font() -> tuple:
    """检测系统可用字体，返回最佳中文字体。"""
    try:
        available = tkfont.families()
    except Exception:
        available = []
    for name in ['MiSans', 'Microsoft YaHei UI', 'PingFang SC',
                 'Noto Sans CJK SC', 'Microsoft YaHei', 'Segoe UI']:
        if name in available:
            return name
    return 'TkDefaultFont'


class PuzzleSplitterApp:
    """拼图分割工具主窗口。"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("拼图分割工具")
        self.root.geometry("1000x700")
        self.root.configure(bg=C_WINDOW_BG)
        self.root.minsize(900, 600)

        # 字体
        self.font_family = _detect_font()
        self.f_title = (self.font_family, 15, 'bold')
        self.f_body = (self.font_family, 13)
        self.f_small = (self.font_family, 11)
        self.f_tiny = (self.font_family, 10)
        self.f_mono = (self.font_family, 12)

        # 原图数据
        self.image_bgr = None
        self.detector = None
        self.display_scale = 1.0
        self.h_lines: list[int] = []
        self.v_lines: list[int] = []
        self.canvas_image = None

        self._setup_style()
        self._build_ui()

    # ------------------------------------------------------------------
    # 样式配置
    # ------------------------------------------------------------------
    def _setup_style(self):
        """配置ttk样式为Apple风格。"""
        style = ttk.Style()
        try:
            # 使用clam主题作为基础（最可定制）
            style.theme_use('clam')
        except Exception:
            pass

        # 全局背景
        style.configure('.', background=C_WINDOW_BG, foreground=C_TEXT,
                        font=self.f_body)

        # Frame
        style.configure('Card.TFrame', background=C_CARD_BG)
        style.configure('Window.TFrame', background=C_WINDOW_BG)
        style.configure('Panel.TFrame', background=C_WINDOW_BG)

        # Label
        style.configure('Title.TLabel', background=C_WINDOW_BG,
                        foreground=C_TEXT, font=self.f_title)
        style.configure('Body.TLabel', background=C_CARD_BG,
                        foreground=C_TEXT, font=self.f_body)
        style.configure('Small.TLabel', background=C_CARD_BG,
                        foreground=C_TEXT_SEC, font=self.f_small)
        style.configure('CardTitle.TLabel', background=C_CARD_BG,
                        foreground=C_TEXT_SEC, font=self.f_tiny)
        style.configure('Status.TLabel', background=C_STATUS_BG,
                        foreground=C_TEXT_SEC, font=self.f_small)
        style.configure('ColTitle.TLabel', background=C_CARD_BG,
                        foreground=C_TEXT_SEC, font=(self.font_family, 10))

        # Radiobutton
        style.configure('TRadiobutton', background=C_CARD_BG,
                        foreground=C_TEXT, font=self.f_body)
        style.map('TRadiobutton',
                  background=[('active', C_CARD_BG)])

        # Scale (滑块)
        style.configure('TScale', background=C_BLUE, troughcolor=C_BTN_SEC_BG)

        # Spinbox
        style.configure('TSpinbox', fieldbackground=C_CARD_BG,
                        foreground=C_TEXT, font=self.f_body,
                        bordercolor=C_CARD_BORDER, focuscolor=C_BLUE)

        # TFrame for toolbar
        style.configure('Toolbar.TFrame', background=C_CARD_BG)

    # ------------------------------------------------------------------
    # UI构建
    # ------------------------------------------------------------------
    def _build_ui(self):
        """构建Apple风格界面。"""
        # ===== 顶部工具栏 =====
        toolbar = tk.Frame(self.root, bg=C_CARD_BG, height=52)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        toolbar.pack_propagate(False)

        # 底部细线
        tk.Frame(toolbar, bg=C_STATUS_BORDER, height=1).pack(
            side=tk.BOTTOM, fill=tk.X)

        tb_inner = tk.Frame(toolbar, bg=C_CARD_BG)
        tb_inner.pack(fill=tk.X, padx=16, pady=8)

        # 打开图片按钮（蓝色主按钮）
        btn_open = tk.Button(tb_inner, text="打开图片", font=self.f_body,
                             fg='white', bg=C_BLUE, activebackground=C_BLUE_DARK,
                             activeforeground='white', relief='flat', bd=0,
                             cursor='hand2', padx=14, pady=4,
                             command=self._on_open_image)
        btn_open.pack(side=tk.LEFT)

        # 居中标题
        title = tk.Label(tb_inner, text="拼图分割工具", font=self.f_title,
                         bg=C_CARD_BG, fg=C_TEXT)
        title.pack(side=tk.LEFT, expand=True, fill=tk.X)
        # 真正居中：用place
        self.root.update_idletasks()
        title.pack_forget()
        title.place(in_=toolbar, relx=0.5, rely=0.5, anchor='center')
        # 右侧占位保持平衡
        tk.Frame(tb_inner, bg=C_CARD_BG, width=90).pack(side=tk.RIGHT)

        # ===== 主区域 =====
        main = tk.Frame(self.root, bg=C_WINDOW_BG)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=14, pady=14)

        # 左侧画布区
        self._build_canvas_area(main)

        # 右侧面板
        self._build_panel(main)

    def _make_card(self, parent, **kw) -> tk.Frame:
        """创建白色圆角卡片Frame。"""
        card = tk.Frame(parent, bg=C_CARD_BG, highlightbackground=C_CARD_BORDER,
                        highlightthickness=1, bd=0, **kw)
        return card

    def _make_btn(self, parent, text, primary=True, **kw) -> tk.Button:
        """创建Apple风格按钮。"""
        if primary:
            bg, ab, fg = C_BLUE, C_BLUE_DARK, 'white'
        else:
            bg, ab, fg = C_BTN_SEC_BG, C_BTN_SEC_DARK, C_TEXT
        return tk.Button(parent, text=text, font=self.f_body, fg=fg, bg=bg,
                         activebackground=ab, activeforeground=fg, relief='flat',
                         bd=0, cursor='hand2', pady=6, **kw)

    def _build_canvas_area(self, parent):
        """构建左侧画布区。"""
        # 白色卡片容器
        card = self._make_card(parent)
        card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 14))

        # 画布
        self.canvas = tk.Canvas(card, bg=C_CANVAS_BG, highlightthickness=0,
                                cursor='crosshair')
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=2, pady=2)

        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Shift-Button-1>", self._on_shift_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)

        # 状态栏
        status_bar = tk.Frame(card, bg=C_STATUS_BG, height=34)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=2, pady=(0, 2))
        status_bar.pack_propagate(False)

        tk.Frame(status_bar, bg=C_STATUS_BORDER, height=1).pack(
            side=tk.TOP, fill=tk.X)

        self.status_var = tk.StringVar(value="请点击「打开图片」加载图片")
        lbl_status = tk.Label(status_bar, textvariable=self.status_var,
                              font=self.f_small, bg=C_STATUS_BG, fg=C_TEXT_SEC)
        lbl_status.pack(side=tk.LEFT, padx=14, pady=8)

        lbl_hint = tk.Label(status_bar,
                            text="左键:水平线  ⇧+左键:垂直线  右键:删除",
                            font=self.f_tiny, bg=C_STATUS_BG, fg=C_TEXT_SEC)
        lbl_hint.pack(side=tk.RIGHT, padx=14, pady=8)

    def _build_panel(self, parent):
        """构建右侧控制面板。"""
        panel = tk.Frame(parent, bg=C_WINDOW_BG, width=268)
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        panel.pack_propagate(False)

        # --- 检测模式卡片 ---
        mode_card = self._make_card(panel)
        mode_card.pack(fill=tk.X, pady=(0, 10), padx=2)
        inner = tk.Frame(mode_card, bg=C_CARD_BG)
        inner.pack(fill=tk.X, padx=14, pady=13)

        tk.Label(inner, text="检测模式", font=self.f_tiny, bg=C_CARD_BG,
                 fg=C_TEXT_SEC).pack(anchor=tk.W)

        # 自动检测
        self.mode_var = tk.StringVar(value="auto")
        rb_auto = ttk.Radiobutton(inner, text="自动检测", variable=self.mode_var,
                                   value="auto", command=self._on_mode_change,
                                   style='TRadiobutton')
        rb_auto.pack(anchor=tk.W, pady=(8, 0))

        # 灵敏度滑块
        sens_frame = tk.Frame(inner, bg=C_CARD_BG)
        sens_frame.pack(fill=tk.X, padx=23, pady=(4, 0))
        tk.Label(sens_frame, text="灵敏度", font=self.f_small, bg=C_CARD_BG,
                 fg=C_TEXT_SEC).pack(side=tk.LEFT)
        self.sens_var = tk.DoubleVar(value=0.05)
        self.sens_scale = ttk.Scale(sens_frame, from_=0.0, to=1.0,
                                    variable=self.sens_var, orient=tk.HORIZONTAL,
                                    command=lambda v: self.sens_label.config(
                                        text=f"{float(v):.2f}"))
        self.sens_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.sens_label = tk.Label(sens_frame, text="0.05", font=self.f_body,
                                    bg=C_CARD_BG, fg=C_TEXT, width=5)
        self.sens_label.pack(side=tk.LEFT)

        # 网格均分
        grid_frame = tk.Frame(inner, bg=C_CARD_BG)
        grid_frame.pack(fill=tk.X, pady=(4, 0))
        rb_grid = ttk.Radiobutton(grid_frame, text="网格均分", variable=self.mode_var,
                                   value="grid", command=self._on_mode_change,
                                   style='TRadiobutton')
        rb_grid.pack(side=tk.LEFT)

        inputs = tk.Frame(grid_frame, bg=C_CARD_BG)
        inputs.pack(side=tk.RIGHT)
        tk.Label(inputs, text="行", font=self.f_small, bg=C_CARD_BG,
                 fg=C_TEXT_SEC).pack(side=tk.LEFT, padx=(0, 2))
        self.rows_var = tk.IntVar(value=1)
        sb_r = ttk.Spinbox(inputs, from_=1, to=20, width=3, textvariable=self.rows_var,
                           font=self.f_body)
        sb_r.pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(inputs, text="列", font=self.f_small, bg=C_CARD_BG,
                 fg=C_TEXT_SEC).pack(side=tk.LEFT, padx=(0, 2))
        self.cols_var = tk.IntVar(value=7)
        sb_c = ttk.Spinbox(inputs, from_=1, to=20, width=3, textvariable=self.cols_var,
                           font=self.f_body)
        sb_c.pack(side=tk.LEFT)

        # 执行检测按钮
        btn_detect = self._make_btn(inner, "执行检测", primary=True)
        btn_detect.pack(fill=tk.X, pady=(10, 0))

        # --- 分割线列表卡片 ---
        list_card = self._make_card(panel)
        list_card.pack(fill=tk.BOTH, expand=True, pady=(0, 10), padx=2)
        list_inner = tk.Frame(list_card, bg=C_CARD_BG)
        list_inner.pack(fill=tk.BOTH, expand=True, padx=14, pady=13)

        tk.Label(list_inner, text="分割线", font=self.f_tiny, bg=C_CARD_BG,
                 fg=C_TEXT_SEC).pack(anchor=tk.W)

        cols = tk.Frame(list_inner, bg=C_CARD_BG)
        cols.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        # 水平线列
        h_col = tk.Frame(cols, bg=C_CARD_BG)
        h_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        tk.Label(h_col, text="水平线 (Y)", font=(self.font_family, 10),
                 bg=C_CARD_BG, fg=C_TEXT_SEC).pack(anchor=tk.W, pady=(0, 3))
        self.h_listbox = tk.Listbox(h_col, bg=C_LIST_BG, fg=C_TEXT,
                                    selectbackground=C_BLUE, selectforeground='white',
                                    relief='flat', bd=0, highlightthickness=0,
                                    font=self.f_mono, activestyle='none', height=6)
        self.h_listbox.pack(fill=tk.BOTH, expand=True)

        # 垂直线列
        v_col = tk.Frame(cols, bg=C_CARD_BG)
        v_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        tk.Label(v_col, text="垂直线 (X)", font=(self.font_family, 10),
                 bg=C_CARD_BG, fg=C_TEXT_SEC).pack(anchor=tk.W, pady=(0, 3))
        self.v_listbox = tk.Listbox(v_col, bg=C_LIST_BG, fg=C_TEXT,
                                    selectbackground=C_BLUE, selectforeground='white',
                                    relief='flat', bd=0, highlightthickness=0,
                                    font=self.f_mono, activestyle='none', height=6)
        self.v_listbox.pack(fill=tk.BOTH, expand=True)

        # 操作按钮
        btns = tk.Frame(list_inner, bg=C_CARD_BG)
        btns.pack(fill=tk.X, pady=(7, 0))
        for text, cmd in [("添加", None), ("删除选中", None), ("清空", None)]:
            b = tk.Button(btns, text=text, font=self.f_small, fg=C_TEXT,
                          bg=C_BTN_SEC_BG, activebackground=C_BTN_SEC_DARK,
                          relief='flat', bd=0, cursor='hand2', pady=4)
            b.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
            if text == "添加":
                b.config(command=lambda: self._on_add_line_dialog())
            elif text == "删除选中":
                b.config(command=self._on_delete_selected)
            elif text == "清空":
                b.config(command=self._on_clear_all)

        # --- 导出卡片 ---
        export_card = self._make_card(panel)
        export_card.pack(fill=tk.X, padx=2)
        export_inner = tk.Frame(export_card, bg=C_CARD_BG)
        export_inner.pack(fill=tk.X, padx=14, pady=10)

        btn_export = self._make_btn(export_inner, "导出分割结果", primary=True)
        btn_export.pack(fill=tk.X, pady=(0, 7))
        btn_marked = self._make_btn(export_inner, "保存标记图片", primary=False)
        btn_marked.pack(fill=tk.X)

        self._on_mode_change()

    # ------------------------------------------------------------------
    # 模式切换
    # ------------------------------------------------------------------
    def _on_mode_change(self):
        """切换检测模式时启用/禁用相关控件。"""
        if self.mode_var.get() == "auto":
            self.sens_scale.state(['!disabled'])
        else:
            self.sens_scale.state(['disabled'])

    # ------------------------------------------------------------------
    # 图片加载与显示
    # ------------------------------------------------------------------
    def _on_open_image(self):
        """打开图片文件。"""
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("所有文件", "*.*")]
        )
        if not path:
            return

        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("错误", f"无法读取图片:\n{path}")
            return

        self.image_bgr = img
        h, w = img.shape[:2]
        self.status_var.set(f"正在分析图片 {w}×{h} ...")

        if max(h, w) > 5000:
            self.status_var.set(f"图片较大 {w}×{h}，特征计算可能需要几秒...")

        def worker():
            try:
                self.detector = SeamDetector(img)
                self.h_lines = []
                self.v_lines = []
                self.root.after(0, lambda: self._after_load(w, h))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "错误", f"图片分析失败:\n{e}"))
                self.root.after(0, lambda: self.status_var.set("图片分析失败"))

        threading.Thread(target=worker, daemon=True).start()

    def _after_load(self, w: int, h: int):
        self._update_display()
        self._update_line_lists()
        self.status_var.set(f"已加载 {w}×{h}，可执行检测或手动添加分割线")

    def _compute_display_scale(self) -> float:
        if self.image_bgr is None:
            return 1.0
        h, w = self.image_bgr.shape[:2]
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        # 留16px边距
        cw -= 32
        ch -= 32
        scale = min(cw / w, ch / h, 1.0)
        return scale if scale > 0 else 1.0

    def _update_display(self):
        """重绘Canvas：图片+分割线。"""
        if self.image_bgr is None:
            return

        self.display_scale = self._compute_display_scale()
        h, w = self.image_bgr.shape[:2]
        new_w = int(w * self.display_scale)
        new_h = int(h * self.display_scale)

        rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS)
        self.canvas_image = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        # 居中显示，留16px边距
        ox = (self.canvas.winfo_width() - new_w) // 2
        oy = (self.canvas.winfo_height() - new_h) // 2
        ox = max(ox, 16)
        oy = max(oy, 16)
        self.canvas.create_image(ox, oy, anchor=tk.NW, image=self.canvas_image)

        # 水平分割线（Apple Red）
        for y in self.h_lines:
            dy = int(y * self.display_scale) + oy
            self.canvas.create_line(ox, dy, ox + new_w, dy,
                                    fill=C_RED, width=2, tags="hline")

        # 垂直分割线（Apple Blue）
        for x in self.v_lines:
            dx = int(x * self.display_scale) + ox
            self.canvas.create_line(dx, oy, dx, oy + new_h,
                                    fill=C_BLUE, width=2, tags="vline")

    def _update_line_lists(self):
        """刷新分割线列表。"""
        self.h_listbox.delete(0, tk.END)
        for y in sorted(self.h_lines):
            self.h_listbox.insert(tk.END, f"y={y}")
        self.v_listbox.delete(0, tk.END)
        for x in sorted(self.v_lines):
            self.v_listbox.insert(tk.END, f"x={x}")

    # ------------------------------------------------------------------
    # 检测
    # ------------------------------------------------------------------
    def _on_detect(self):
        if self.image_bgr is None:
            messagebox.showwarning("提示", "请先打开图片")
            return
        if self.detector is None:
            messagebox.showwarning("提示", "图片仍在分析中，请稍候")
            return
        if self.mode_var.get() == "auto":
            self._run_auto_detect()
        else:
            self._run_grid_detect()

    def _run_auto_detect(self):
        sens = self.sens_var.get()
        self.status_var.set("正在检测分割线...")

        def worker():
            try:
                h_lines = self.detector.detect_horizontal(sens)
                v_lines = self.detector.detect_vertical(sens)
                self.root.after(0, lambda: self._apply_detect_result(h_lines, v_lines))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "错误", f"检测失败:\n{e}"))
                self.root.after(0, lambda: self.status_var.set("检测失败"))

        threading.Thread(target=worker, daemon=True).start()

    def _run_grid_detect(self):
        h, w = self.image_bgr.shape[:2]
        rows = max(1, int(self.rows_var.get()))
        cols = max(1, int(self.cols_var.get()))
        h_lines = [int(h * i / rows) for i in range(1, rows)]
        v_lines = [int(w * i / cols) for i in range(1, cols)]
        self._apply_detect_result(h_lines, v_lines)

    def _apply_detect_result(self, h_lines: list, v_lines: list):
        self.h_lines = sorted(set(int(y) for y in h_lines))
        self.v_lines = sorted(set(int(x) for x in v_lines))
        self._update_display()
        self._update_line_lists()
        self.status_var.set(
            f"检测完成: {len(self.h_lines)}条水平线, {len(self.v_lines)}条垂直线")

    # ------------------------------------------------------------------
    # 手动编辑分割线
    # ------------------------------------------------------------------
    def _on_left_click(self, event):
        self._add_line_at(event, 'h')

    def _on_shift_left_click(self, event):
        self._add_line_at(event, 'v')

    def _add_line_at(self, event, kind: str):
        if self.image_bgr is None:
            messagebox.showwarning("提示", "请先打开图片")
            return

        oy = event.y / self.display_scale
        ox = event.x / self.display_scale
        h, w = self.image_bgr.shape[:2]

        # 减去画布偏移
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        img_w = int(w * self.display_scale)
        img_h = int(h * self.display_scale)
        off_x = max((canvas_w - img_w) // 2, 16)
        off_y = max((canvas_h - img_h) // 2, 16)
        oy = (event.y - off_y) / self.display_scale
        ox = (event.x - off_x) / self.display_scale

        if kind == 'h':
            y = int(round(oy))
            y = max(1, min(h - 1, y))
            if y not in self.h_lines:
                self.h_lines.append(y)
                self.h_lines.sort()
        else:
            x = int(round(ox))
            x = max(1, min(w - 1, x))
            if x not in self.v_lines:
                self.v_lines.append(x)
                self.v_lines.sort()

        self._update_display()
        self._update_line_lists()

    def _on_right_click(self, event):
        if self.image_bgr is None:
            return

        oy = event.y / self.display_scale
        ox = event.x / self.display_scale

        nearest_h = None
        if self.h_lines:
            nearest_h = min(self.h_lines, key=lambda y: abs(y - oy))
            dist_h = abs(nearest_h - oy)
        else:
            dist_h = float('inf')

        nearest_v = None
        if self.v_lines:
            nearest_v = min(self.v_lines, key=lambda x: abs(x - ox))
            dist_v = abs(nearest_v - ox)
        else:
            dist_v = float('inf')

        if dist_h == float('inf') and dist_v == float('inf'):
            return

        if dist_h <= dist_v:
            self.h_lines.remove(nearest_h)
        else:
            self.v_lines.remove(nearest_v)

        self._update_display()
        self._update_line_lists()

    def _on_add_line_dialog(self):
        """通过对话框添加分割线。"""
        if self.image_bgr is None:
            messagebox.showwarning("提示", "请先打开图片")
            return

        val = simpledialog.askinteger("添加分割线", "请输入坐标值:")
        if val is None:
            return

        # 判断添加水平还是垂直（根据哪个列表有选中）
        h_sel = self.h_listbox.curselection()
        v_sel = self.v_listbox.curselection()
        if v_sel and not h_sel:
            kind = 'v'
        else:
            kind = 'h'

        h, w = self.image_bgr.shape[:2]
        if kind == 'h':
            if val < 1 or val >= h:
                messagebox.showwarning("超出范围", f"y坐标必须在 1 到 {h-1} 之间")
                return
            if val not in self.h_lines:
                self.h_lines.append(val)
                self.h_lines.sort()
        else:
            if val < 1 or val >= w:
                messagebox.showwarning("超出范围", f"x坐标必须在 1 到 {w-1} 之间")
                return
            if val not in self.v_lines:
                self.v_lines.append(val)
                self.v_lines.sort()

        self._update_display()
        self._update_line_lists()

    def _on_delete_selected(self):
        """删除列表中选中的分割线。"""
        h_sel = self.h_listbox.curselection()
        v_sel = self.v_listbox.curselection()
        if h_sel:
            idx = h_sel[0]
            if 0 <= idx < len(self.h_lines):
                self.h_lines.pop(idx)
        elif v_sel:
            idx = v_sel[0]
            if 0 <= idx < len(self.v_lines):
                self.v_lines.pop(idx)
        else:
            messagebox.showwarning("提示", "请先在列表中选中一条分割线")
            return
        self._update_display()
        self._update_line_lists()

    def _on_clear_all(self):
        """清空所有分割线。"""
        self.h_lines = []
        self.v_lines = []
        self._update_display()
        self._update_line_lists()

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    def _on_export(self):
        if self.image_bgr is None:
            messagebox.showwarning("提示", "请先打开图片")
            return
        if self.detector is None:
            messagebox.showwarning("提示", "图片仍在分析中")
            return

        if not self.h_lines and not self.v_lines:
            if not messagebox.askyesno("提示", "当前无分割线，将导出整张图片。继续？"):
                return

        out_dir = filedialog.askdirectory(title="选择导出目录")
        if not out_dir:
            return

        try:
            pieces = self.detector.split(self.h_lines, self.v_lines)
            for i, piece in enumerate(pieces, start=1):
                fname = f"split_{i:02d}.png"
                fpath = os.path.join(out_dir, fname)
                ok, buf = cv2.imencode(".png", piece)
                if ok:
                    buf.tofile(fpath)
            messagebox.showinfo("成功", f"已导出 {len(pieces)} 张子图到:\n{out_dir}")
            self.status_var.set(f"已导出 {len(pieces)} 张子图")
        except Exception as e:
            messagebox.showerror("导出失败", str(e))

    def _on_save_marked(self):
        if self.image_bgr is None:
            messagebox.showwarning("提示", "请先打开图片")
            return

        path = filedialog.asksaveasfilename(
            title="保存标记图片", defaultextension=".png",
            filetypes=[("PNG图片", "*.png"), ("JPEG图片", "*.jpg")])
        if not path:
            return

        marked = self.image_bgr.copy()
        for y in self.h_lines:
            cv2.line(marked, (0, y), (marked.shape[1], y), (0, 59, 255), 2)
        for x in self.v_lines:
            cv2.line(marked, (x, 0), (x, marked.shape[0]), (255, 122, 0), 2)

        try:
            ext = os.path.splitext(path)[1] or ".png"
            ok, buf = cv2.imencode(ext, marked)
            if ok:
                buf.tofile(path)
                messagebox.showinfo("成功", f"已保存标记图片到:\n{path}")
                self.status_var.set("标记图片已保存")
            else:
                messagebox.showerror("失败", "编码失败，请检查文件扩展名")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))


def main():
    """程序入口。"""
    root = tk.Tk()
    app = PuzzleSplitterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
