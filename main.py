"""拼图分割工具 - GUI主程序 (Apple风格UI v2)。

基于Tkinter，采用Apple原生应用设计语言。
v2 改进：全圆角按钮、优化布局防止截断、Apple设计规范对齐。
支持：自动检测、网格均分、拖动分割线、Delete键删除。
"""

from __future__ import annotations

import math
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
    raise SystemExit("Tkinter is required.") from e

from splitter import SeamDetector


# ============================================================
# Apple风格配色常量
# ============================================================
C_WINDOW_BG = '#F5F5F7'
C_CARD_BG = '#FFFFFF'
C_CARD_BORDER = '#E5E5EA'
C_CANVAS_BG = '#F0F0F2'
C_BLUE = '#007AFF'
C_BLUE_DARK = '#0051D5'
C_BLUE_HOVER = '#0062CC'
C_RED = '#FF3B30'
C_GREEN = '#34C759'           # 选中线
C_TEXT = '#1D1D1F'
C_TEXT_SEC = '#86868B'
C_BTN_SEC_BG = '#E5E5EA'
C_BTN_SEC_DARK = '#C7C7CC'
C_BTN_SEC_HOVER = '#D1D1D6'
C_LIST_BG = '#F5F5F7'
C_STATUS_BG = '#FAFAFA'
C_STATUS_BORDER = '#E5E5EA'

# 圆角半径常量
R_BTN = 8          # 按钮圆角
R_CARD = 12        # 卡片圆角(通过模拟)
R_SMALL = 6        # 小控件圆角


def _detect_font() -> str:
    try:
        available = tkfont.families()
    except Exception:
        available = []
    for name in ['MiSans', 'Microsoft YaHei UI', 'PingFang SC',
                 'Noto Sans CJK SC', 'Microsoft YaHei', 'Segoe UI']:
        if name in available:
            return name
    return 'TkDefaultFont'


class RoundedButton(tk.Canvas):
    """Apple风格圆角按钮（基于Canvas实现）。

    使用Canvas绘制圆角矩形（平直四边 + 圆角）作为按钮背景，
    支持主色(primary)和次要色(secondary)两种样式，
    自带hover高亮效果和点击回调。
    绑定 <Configure> 在控件映射/尺寸变化时按真实尺寸重绘，
    避免初始未按实际宽度绘制导致内容不可见。
    """

    def __init__(self, parent, text: str, command=None,
                 primary: bool = True,
                 font=None, height: int = 32,
                 padding_x: int = 16, radius: int = R_BTN,
                 **kw):
        # 计算实际尺寸：文字宽度 + 左右padding
        self._font = font or ('Segoe UI', 11)
        self._text = text
        self._primary = primary
        self._radius = radius
        self._command = command
        self._padding_x = padding_x

        # 颜色配置
        if primary:
            self._bg_color = C_BLUE
            self._fg_color = 'white'
            self._hover_bg = C_BLUE_HOVER
            self._active_bg = C_BLUE_DARK
            self._outline = C_BLUE_DARK
        else:
            self._bg_color = C_BTN_SEC_BG
            self._fg_color = C_TEXT
            self._hover_bg = C_BTN_SEC_HOVER
            self._active_bg = C_BTN_SEC_DARK
            self._outline = C_BTN_SEC_DARK

        # 估算宽度
        test_label = tk.Label(parent, text=text, font=self._font)
        text_w = test_label.winfo_reqwidth()
        test_label.destroy()
        width = text_w + padding_x * 2 + 8
        self._height = height

        super().__init__(
            parent, width=width, height=height,
            highlightthickness=0, bd=0,
            bg=C_CARD_BG, cursor='hand2', **kw
        )

        self._rect_id = None
        self._text_id = None
        self._current_color = self._bg_color
        self._draw_button(self._bg_color)

        # 事件绑定：<Configure> 用于控件映射/缩放后按真实尺寸重绘
        self.bind('<Configure>', self._on_configure)
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
        self.bind('<Button-1>', self._on_click)

    @staticmethod
    def _round_rect_points(x1, y1, x2, y2, r):
        """生成圆角矩形的多边形顶点（四角圆弧采样 + 四边直线）。"""
        r = max(0.0, min(float(r), (x2 - x1) / 2.0, (y2 - y1) / 2.0))
        pts = []
        n = 12  # 每个圆角的采样点数，越大越平滑
        # 四个圆角：(圆心x, 圆心y, 起始角度°)，顺时针：右上→右下→左下→左上
        corners = [
            (x2 - r, y1 + r, -90.0),
            (x2 - r, y2 - r, 0.0),
            (x1 + r, y2 - r, 90.0),
            (x1 + r, y1 + r, 180.0),
        ]
        for cx, cy, a0 in corners:
            for i in range(n + 1):
                ang = math.radians(a0 + 90.0 * i / n)
                pts.append(cx + r * math.cos(ang))
                pts.append(cy + r * math.sin(ang))
        return pts

    def _draw_button(self, bg_color):
        """绘制按钮外观（圆角矩形 + 文字）。"""
        self._current_color = bg_color
        self.delete('all')
        w = self.winfo_width()
        if w < 2:
            w = int(self['width'])
        h = self.winfo_height()
        if h < 2:
            h = self._height
        if w < 2 or h < 2:
            return
        r = min(self._radius, h // 2, w // 2)

        pts = self._round_rect_points(0, 0, w, h, r)
        self._rect_id = self.create_polygon(
            pts, smooth=False,
            fill=bg_color, outline=self._outline, width=1
        )

        # 文字居中
        self._text_id = self.create_text(
            w // 2, h // 2,
            text=self._text, fill=self._fg_color,
            font=self._font, anchor='center'
        )

    def _on_configure(self, event):
        """控件映射/尺寸变化时按当前颜色重绘。"""
        self._draw_button(self._current_color)

    def _on_enter(self, event):
        self._draw_button(self._hover_bg)

    def _on_leave(self, event):
        self._draw_button(self._bg_color)

    def _on_click(self, event):
        self._draw_button(self._active_bg)
        self.after(100, lambda: self._draw_button(self._hover_bg))
        if self._command:
            self._command()

    def config(self, **kw):
        super().config(**kw)
        if 'text' in kw:
            self._text = kw['text']
            self._draw_button(self._current_color)


class PuzzleSplitterApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Puzzle Splitter")
        self.root.geometry("1150x750")
        self.root.configure(bg=C_WINDOW_BG)
        self.root.minsize(1000, 680)

        # 字体
        self.font_family = _detect_font()
        self.f_title = (self.font_family, 15, 'bold')
        self.f_body = (self.font_family, 13)
        self.f_small = (self.font_family, 11)
        self.f_tiny = (self.font_family, 10)
        self.f_mono = (self.font_family, 12)

        # 数据
        self.image_bgr = None
        self.detector = None
        self.display_scale = 1.0
        self.h_lines: list[int] = []
        self.v_lines: list[int] = []
        self.canvas_image = None

        # 交互状态
        self.selected_line = None       # (kind, value) 或 None
        self.dragging_line = None       # (kind, index) 或 None
        self._resize_after_id = None
        self.image_name = ""            # 源图片文件名（不含扩展名），用于导出命名

        self._setup_style()
        self._build_ui()

        # 全局键盘绑定
        self.root.bind("<Delete>", self._on_delete_key)
        self.root.bind("<BackSpace>", self._on_delete_key)

    # ------------------------------------------------------------------
    # 样式
    # ------------------------------------------------------------------
    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass
        style.configure('.', background=C_WINDOW_BG, foreground=C_TEXT,
                        font=self.f_body)
        style.configure('Card.TFrame', background=C_CARD_BG)
        style.configure('Window.TFrame', background=C_WINDOW_BG)
        style.configure('Title.TLabel', background=C_WINDOW_BG,
                        foreground=C_TEXT, font=self.f_title)
        style.configure('TRadiobutton', background=C_CARD_BG,
                        foreground=C_TEXT, font=self.f_body)
        style.map('TRadiobutton', background=[('active', C_CARD_BG)])
        style.configure('TScale', background=C_BLUE, troughcolor=C_BTN_SEC_BG)
        style.configure('TSpinbox', fieldbackground=C_CARD_BG,
                        foreground=C_TEXT, font=self.f_body,
                        bordercolor=C_CARD_BORDER)

    # ------------------------------------------------------------------
    # UI构建
    # ------------------------------------------------------------------
    def _build_ui(self):
        # ===== 顶部工具栏 =====
        toolbar = tk.Frame(self.root, bg=C_CARD_BG, height=56)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        toolbar.pack_propagate(False)
        # 底部分隔线
        tk.Frame(toolbar, bg=C_CARD_BORDER, height=1).pack(
            side=tk.BOTTOM, fill=tk.X)

        tb_inner = tk.Frame(toolbar, bg=C_CARD_BG)
        tb_inner.pack(fill=tk.X, padx=18, pady=10)

        # Open Image 按钮（圆角）
        btn_open = RoundedButton(tb_inner, text="Open Image",
                                 font=self.f_body,
                                 command=self._on_open_image,
                                 primary=True, height=34,
                                 padding_x=18, radius=R_BTN)
        btn_open.pack(side=tk.LEFT)

        title = tk.Label(toolbar, text="Puzzle Splitter", font=self.f_title,
                         bg=C_CARD_BG, fg=C_TEXT)
        title.place(relx=0.5, rely=0.5, anchor='center')

        # ===== 主区域 =====
        main = tk.Frame(self.root, bg=C_WINDOW_BG)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=14, pady=12)
        self._build_canvas_area(main)
        self._build_panel(main)

    def _make_card(self, parent, **kw) -> tk.Frame:
        """创建Apple风格的白色卡片容器。"""
        return tk.Frame(parent, bg=C_CARD_BG,
                        highlightbackground=C_CARD_BORDER,
                        highlightthickness=1, bd=0, **kw)

    def _make_rounded_btn(self, parent, text: str, primary: bool = True,
                           command=None, height: int = 34, **kw):
        """创建圆角按钮的便捷方法。"""
        return RoundedButton(parent, text=text, command=command,
                             primary=primary, font=self.f_body,
                             height=height, radius=R_BTN, **kw)

    def _make_small_btn(self, parent, text: str, command=None, **kw):
        """小型圆角按钮（用于Add/Delete/Clear等）。"""
        return RoundedButton(parent, text=text, command=command,
                             primary=False, font=self.f_small,
                             height=28, radius=R_SMALL,
                             padding_x=10, **kw)

    def _build_canvas_area(self, parent):
        """左侧画布区域。"""
        card = self._make_card(parent)
        card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        self.canvas = tk.Canvas(card, bg=C_CANVAS_BG, highlightthickness=0,
                                cursor='crosshair')
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=3, pady=3)

        # 事件绑定
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Shift-Button-1>", self._on_shift_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # 状态栏
        status_bar = tk.Frame(card, bg=C_STATUS_BG, height=36)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=3, pady=(0, 3))
        status_bar.pack_propagate(False)
        tk.Frame(status_bar, bg=C_STATUS_BORDER, height=1).pack(
            side=tk.TOP, fill=tk.X)

        self.status_var = tk.StringVar(value="Click 'Open Image' to load")
        tk.Label(status_bar, textvariable=self.status_var,
                 font=self.f_small, bg=C_STATUS_BG, fg=C_TEXT_SEC
                 ).pack(side=tk.LEFT, padx=14, pady=9)
        tk.Label(status_bar,
                 text="Click: add H | Shift+Click: add V | Drag: move | Del: delete",
                 font=self.f_tiny, bg=C_STATUS_BG, fg=C_TEXT_SEC
                 ).pack(side=tk.RIGHT, padx=14, pady=9)

    def _build_panel(self, parent):
        """右侧控制面板。"""
        panel = tk.Frame(parent, bg=C_WINDOW_BG, width=290)
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        panel.pack_propagate(False)

        # --- 检测模式卡片 ---
        mode_card = self._make_card(panel)
        mode_card.pack(fill=tk.X, pady=(0, 10), padx=2)
        inner = tk.Frame(mode_card, bg=C_CARD_BG)
        inner.pack(fill=tk.X, padx=16, pady=14)

        tk.Label(inner, text="Detection Mode", font=self.f_tiny,
                 bg=C_CARD_BG, fg=C_TEXT_SEC).pack(anchor=tk.W)

        self.mode_var = tk.StringVar(value="auto")
        ttk.Radiobutton(inner, text="Auto Detect", variable=self.mode_var,
                        value="auto", command=self._on_mode_change
                        ).pack(anchor=tk.W, pady=(8, 0))

        # 灵敏度滑块
        sens_frame = tk.Frame(inner, bg=C_CARD_BG)
        sens_frame.pack(fill=tk.X, padx=(24, 0), pady=(6, 0))
        tk.Label(sens_frame, text="Sensitivity", font=self.f_small,
                 bg=C_CARD_BG, fg=C_TEXT_SEC).pack(side=tk.LEFT)
        self.sens_var = tk.DoubleVar(value=0.05)
        self.sens_scale = ttk.Scale(sens_frame, from_=0.0, to=1.0,
                                    variable=self.sens_var, orient=tk.HORIZONTAL,
                                    command=lambda v: self.sens_label.config(
                                        text=f"{float(v):.2f}"))
        self.sens_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.sens_label = tk.Label(sens_frame, text="0.05", font=self.f_body,
                                   bg=C_CARD_BG, fg=C_TEXT, width=5,
                                   anchor='w')
        self.sens_label.pack(side=tk.LEFT)

        # Max H / Max V
        exp_frame1 = tk.Frame(inner, bg=C_CARD_BG)
        exp_frame1.pack(fill=tk.X, padx=(24, 0), pady=(6, 0))
        tk.Label(exp_frame1, text="Max H", font=self.f_small, bg=C_CARD_BG,
                 fg=C_TEXT_SEC).pack(side=tk.LEFT)
        self.max_h_var = tk.IntVar(value=0)
        self.max_h_spin = ttk.Spinbox(exp_frame1, from_=0, to=50, width=4,
                                      textvariable=self.max_h_var,
                                      font=self.f_body)
        self.max_h_spin.pack(side=tk.LEFT, padx=(6, 8))
        tk.Label(exp_frame1, text="(0=auto)", font=self.f_tiny, bg=C_CARD_BG,
                 fg=C_TEXT_SEC).pack(side=tk.LEFT)

        exp_frame2 = tk.Frame(inner, bg=C_CARD_BG)
        exp_frame2.pack(fill=tk.X, padx=(24, 0), pady=(4, 0))
        tk.Label(exp_frame2, text="Max V", font=self.f_small, bg=C_CARD_BG,
                 fg=C_TEXT_SEC).pack(side=tk.LEFT)
        self.max_v_var = tk.IntVar(value=0)
        self.max_v_spin = ttk.Spinbox(exp_frame2, from_=0, to=50, width=4,
                                      textvariable=self.max_v_var,
                                      font=self.f_body)
        self.max_v_spin.pack(side=tk.LEFT, padx=(6, 8))
        tk.Label(exp_frame2, text="(0=auto)", font=self.f_tiny, bg=C_CARD_BG,
                 fg=C_TEXT_SEC).pack(side=tk.LEFT)

        # Grid Split 模式
        ttk.Radiobutton(inner, text="Grid Split", variable=self.mode_var,
                        value="grid", command=self._on_mode_change
                        ).pack(anchor=tk.W, pady=(10, 0))

        grid_inputs = tk.Frame(inner, bg=C_CARD_BG)
        grid_inputs.pack(fill=tk.X, padx=(24, 0), pady=(6, 0))
        tk.Label(grid_inputs, text="Rows", font=self.f_small, bg=C_CARD_BG,
                 fg=C_TEXT_SEC).pack(side=tk.LEFT, padx=(0, 4))
        self.rows_var = tk.IntVar(value=1)
        ttk.Spinbox(grid_inputs, from_=1, to=20, width=4,
                    textvariable=self.rows_var, font=self.f_body
                    ).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(grid_inputs, text="Cols", font=self.f_small, bg=C_CARD_BG,
                 fg=C_TEXT_SEC).pack(side=tk.LEFT, padx=(0, 4))
        self.cols_var = tk.IntVar(value=4)
        ttk.Spinbox(grid_inputs, from_=1, to=20, width=4,
                    textvariable=self.cols_var, font=self.f_body
                    ).pack(side=tk.LEFT)
        tk.Label(grid_inputs, text="(1=no split)", font=self.f_tiny,
                 bg=C_CARD_BG, fg=C_TEXT_SEC
                 ).pack(side=tk.LEFT, padx=(6, 0))

        # Run Detection 按钮（主色调圆角）
        self.btn_detect = self._make_rounded_btn(
            inner, "Run Detection", primary=True,
            command=self._on_detect, height=36
        )
        self.btn_detect.pack(fill=tk.X, pady=(12, 0))

        # --- 分割线列表卡片 ---
        list_card = self._make_card(panel)
        list_card.pack(fill=tk.BOTH, expand=True, pady=(0, 10), padx=2)
        list_inner = tk.Frame(list_card, bg=C_CARD_BG)
        list_inner.pack(fill=tk.BOTH, expand=True, padx=16, pady=14)

        tk.Label(list_inner, text="Split Lines", font=self.f_tiny,
                 bg=C_CARD_BG, fg=C_TEXT_SEC).pack(anchor=tk.W)

        cols = tk.Frame(list_inner, bg=C_CARD_BG)
        cols.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        # 水平线列
        h_col = tk.Frame(cols, bg=C_CARD_BG)
        h_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        tk.Label(h_col, text="Horizontal (Y)", font=(self.font_family, 10),
                 bg=C_CARD_BG, fg=C_TEXT_SEC).pack(anchor=tk.W, pady=(0, 4))
        self.h_listbox = tk.Listbox(h_col, bg=C_LIST_BG, fg=C_TEXT,
                                    selectbackground=C_BLUE,
                                    selectforeground='white',
                                    relief='flat', bd=0, highlightthickness=0,
                                    font=self.f_mono, activestyle='none',
                                    height=6, width=9)
        self.h_listbox.pack(fill=tk.BOTH, expand=True)
        self.h_listbox.bind('<<ListboxSelect>>', self._on_h_list_select)

        # 垂直线列
        v_col = tk.Frame(cols, bg=C_CARD_BG)
        v_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        tk.Label(v_col, text="Vertical (X)", font=(self.font_family, 10),
                 bg=C_CARD_BG, fg=C_TEXT_SEC).pack(anchor=tk.W, pady=(0, 4))
        self.v_listbox = tk.Listbox(v_col, bg=C_LIST_BG, fg=C_TEXT,
                                    selectbackground=C_BLUE,
                                    selectforeground='white',
                                    relief='flat', bd=0, highlightthickness=0,
                                    font=self.f_mono, activestyle='none',
                                    height=6, width=9)
        self.v_listbox.pack(fill=tk.BOTH, expand=True)
        self.v_listbox.bind('<<ListboxSelect>>', self._on_v_list_select)

        # 操作按钮行（圆角小按钮）
        btns = tk.Frame(list_inner, bg=C_CARD_BG)
        btns.pack(fill=tk.X, pady=(8, 0))
        for text, cmd in [("Add", self._on_add_line_dialog),
                          ("Delete", self._on_delete_selected),
                          ("Clear", self._on_clear_all)]:
            self._make_small_btn(btns, text=text, command=cmd
                                ).pack(side=tk.LEFT, expand=True,
                                       fill=tk.X, padx=2)

        # --- 导出卡片（确保完整显示）---
        export_card = self._make_card(panel)
        export_card.pack(fill=tk.X, padx=2, pady=(0, 4))  # 底部留一点边距
        export_inner = tk.Frame(export_card, bg=C_CARD_BG)
        export_inner.pack(fill=tk.X, padx=16, pady=12)

        self.btn_export = self._make_rounded_btn(
            export_inner, "Export Results", primary=True,
            command=self._on_export, height=36
        )
        self.btn_export.pack(fill=tk.X, pady=(0, 8))

        self.btn_save = self._make_rounded_btn(
            export_inner, "Save Marked Image", primary=False,
            command=self._on_save_marked, height=32
        )
        self.btn_save.pack(fill=tk.X)

        self._on_mode_change()

    # ------------------------------------------------------------------
    # 模式切换
    # ------------------------------------------------------------------
    def _on_mode_change(self):
        is_auto = self.mode_var.get() == "auto"
        if is_auto:
            self.sens_scale.state(['!disabled'])
            self.max_h_spin.state(['!disabled'])
            self.max_v_spin.state(['!disabled'])
        else:
            self.sens_scale.state(['disabled'])
            self.max_h_spin.state(['disabled'])
            self.max_v_spin.state(['disabled'])

    # ------------------------------------------------------------------
    # 图片加载与显示
    # ------------------------------------------------------------------
    def _on_open_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("All Files", "*.*")]
        )
        if not path:
            return

        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", f"Cannot read image:\n{path}")
            return

        self.image_bgr = img
        self.image_name = os.path.splitext(os.path.basename(path))[0]
        self.selected_line = None
        self.dragging_line = None
        h, w = img.shape[:2]
        self.status_var.set(f"Analyzing {w}x{h} ...")

        if max(h, w) > 5000:
            self.status_var.set(f"Large image {w}x{h}, may take a few seconds...")

        def worker():
            try:
                self.detector = SeamDetector(img)
                self.h_lines = []
                self.v_lines = []
                self.root.after(0, lambda: self._after_load(w, h))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Image analysis failed:\n{e}"))
                self.root.after(0, lambda: self.status_var.set("Analysis failed"))

        threading.Thread(target=worker, daemon=True).start()

    def _after_load(self, w: int, h: int):
        self.root.update_idletasks()
        self._update_display()
        self._update_line_lists()
        self.status_var.set(f"Loaded {w}x{h}. Run detection or add lines manually.")

    def _compute_display_scale(self) -> float:
        if self.image_bgr is None:
            return 1.0
        h, w = self.image_bgr.shape[:2]
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        cw -= 32
        ch -= 32
        scale = min(cw / w, ch / h, 1.0)
        return scale if scale > 0 else 1.0

    def _get_image_offset(self) -> tuple:
        """返回画布上图片的 (ox, oy) 偏移量和缩放后的尺寸。"""
        if self.image_bgr is None:
            return 16, 16, 0, 0
        h, w = self.image_bgr.shape[:2]
        new_w = int(w * self.display_scale)
        new_h = int(h * self.display_scale)
        ox = (self.canvas.winfo_width() - new_w) // 2
        oy = (self.canvas.winfo_height() - new_h) // 2
        return max(ox, 16), max(oy, 16), new_w, new_h

    def _update_display(self):
        """重绘Canvas：图片 + 分割线 + 端点标记。"""
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
        ox, oy, _, _ = self._get_image_offset()
        self.canvas.create_image(ox, oy, anchor=tk.NW, image=self.canvas_image)

        # 水平分割线
        for y in self.h_lines:
            dy = int(y * self.display_scale) + oy
            is_sel = self.selected_line == ('h', y)
            color = C_GREEN if is_sel else C_RED
            lw = 4 if is_sel else 3
            self.canvas.create_line(ox, dy, ox + new_w, dy,
                                    fill=color, width=lw, tags="hline")
            r = 5 if is_sel else 4
            self.canvas.create_rectangle(ox - r, dy - r, ox + r, dy + r,
                                         fill=color, outline='white', width=1)
            self.canvas.create_rectangle(ox + new_w - r, dy - r,
                                         ox + new_w + r, dy + r,
                                         fill=color, outline='white', width=1)

        # 垂直分割线
        for x in self.v_lines:
            dx = int(x * self.display_scale) + ox
            is_sel = self.selected_line == ('v', x)
            color = C_GREEN if is_sel else C_BLUE
            lw = 4 if is_sel else 3
            self.canvas.create_line(dx, oy, dx, oy + new_h,
                                    fill=color, width=lw, tags="vline")
            r = 5 if is_sel else 4
            self.canvas.create_rectangle(dx - r, oy - r, dx + r, oy + r,
                                         fill=color, outline='white', width=1)
            self.canvas.create_rectangle(dx - r, oy + new_h - r,
                                         dx + r, oy + new_h + r,
                                         fill=color, outline='white', width=1)

    def _update_line_lists(self):
        self.h_listbox.delete(0, tk.END)
        for y in sorted(self.h_lines):
            self.h_listbox.insert(tk.END, f"y={y}")
        self.v_listbox.delete(0, tk.END)
        for x in sorted(self.v_lines):
            self.v_listbox.insert(tk.END, f"x={x}")

    # ------------------------------------------------------------------
    # 画布事件：点击、拖拽、删除
    # ------------------------------------------------------------------
    def _on_canvas_resize(self, event):
        if self.image_bgr is None:
            return
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(200, self._update_display)

    def _find_nearest_line(self, event, kind: str, threshold: int = 12):
        """找到离点击位置最近的线条，返回 (index, distance) 或 (None, inf)。"""
        if self.image_bgr is None:
            return None, float('inf')

        ox, oy, _, _ = self._get_image_offset()
        lines = self.h_lines if kind == 'h' else self.v_lines
        click = event.y if kind == 'h' else event.x
        offset = oy if kind == 'h' else ox

        nearest_idx = None
        min_dist = threshold
        for i, val in enumerate(lines):
            canvas_pos = int(val * self.display_scale) + offset
            dist = abs(canvas_pos - click)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        return nearest_idx, min_dist

    def _on_left_click(self, event):
        self._handle_click(event, 'h')

    def _on_shift_left_click(self, event):
        self._handle_click(event, 'v')

    def _handle_click(self, event, default_kind: str):
        """点击：先检查是否点在已有线附近（选中+拖拽），否则添加新线。"""
        if self.image_bgr is None:
            messagebox.showwarning("Tip", "Please open an image first")
            return

        h_idx, h_dist = self._find_nearest_line(event, 'h')
        v_idx, v_dist = self._find_nearest_line(event, 'v')

        if h_idx is not None or v_idx is not None:
            if h_dist <= v_dist:
                kind, idx = 'h', h_idx
                self.selected_line = ('h', self.h_lines[idx])
            else:
                kind, idx = 'v', v_idx
                self.selected_line = ('v', self.v_lines[idx])
            self.dragging_line = (kind, idx)
            self._update_display()
            val = self.selected_line[1]
            self.status_var.set(
                f"Selected: {kind}={val} (drag to move, Del to delete)")
        else:
            self._add_line_at(event, default_kind)

    def _on_drag(self, event):
        """拖拽分割线。"""
        if self.dragging_line is None or self.image_bgr is None:
            return

        kind, idx = self.dragging_line
        h, w = self.image_bgr.shape[:2]
        ox, oy, _, _ = self._get_image_offset()

        if kind == 'h':
            new_y = int(round((event.y - oy) / self.display_scale))
            new_y = max(1, min(h - 1, new_y))
            self.h_lines[idx] = new_y
            self.selected_line = ('h', new_y)
            self.status_var.set(f"Dragging: y={new_y}")
        else:
            new_x = int(round((event.x - ox) / self.display_scale))
            new_x = max(1, min(w - 1, new_x))
            self.v_lines[idx] = new_x
            self.selected_line = ('v', new_x)
            self.status_var.set(f"Dragging: x={new_x}")

        self._update_display()
        self._update_line_lists()

    def _on_release(self, event):
        """释放鼠标：结束拖拽，重排线条。"""
        if self.dragging_line is not None:
            self.h_lines.sort()
            self.v_lines.sort()
            self.dragging_line = None
            self._update_display()
            self._update_line_lists()

    def _on_delete_key(self, event):
        """Delete/BackSpace键删除选中的线。"""
        widget = self.root.focus_get()
        if widget and isinstance(widget, (tk.Entry, ttk.Entry, tk.Spinbox,
                                           ttk.Spinbox, tk.Listbox)):
            return
        if self.selected_line is None:
            return

        kind, val = self.selected_line
        if kind == 'h' and val in self.h_lines:
            self.h_lines.remove(val)
        elif kind == 'v' and val in self.v_lines:
            self.v_lines.remove(val)

        self.selected_line = None
        self.dragging_line = None
        self._update_display()
        self._update_line_lists()
        self.status_var.set("Line deleted")

    def _on_h_list_select(self, event):
        sel = self.h_listbox.curselection()
        if sel and 0 <= sel[0] < len(self.h_lines):
            sorted_h = sorted(self.h_lines)
            self.selected_line = ('h', sorted_h[sel[0]])
            self._update_display()

    def _on_v_list_select(self, event):
        sel = self.v_listbox.curselection()
        if sel and 0 <= sel[0] < len(self.v_lines):
            sorted_v = sorted(self.v_lines)
            self.selected_line = ('v', sorted_v[sel[0]])
            self._update_display()

    # ------------------------------------------------------------------
    # 手动添加分割线
    # ------------------------------------------------------------------
    def _add_line_at(self, event, kind: str):
        if self.image_bgr is None:
            messagebox.showwarning("Tip", "Please open an image first")
            return

        h, w = self.image_bgr.shape[:2]
        ox, oy, _, _ = self._get_image_offset()

        if kind == 'h':
            y = int(round((event.y - oy) / self.display_scale))
            y = max(1, min(h - 1, y))
            if y not in self.h_lines:
                self.h_lines.append(y)
                self.h_lines.sort()
            self.selected_line = ('h', y)
        else:
            x = int(round((event.x - ox) / self.display_scale))
            x = max(1, min(w - 1, x))
            if x not in self.v_lines:
                self.v_lines.append(x)
                self.v_lines.sort()
            self.selected_line = ('v', x)

        self._update_display()
        self._update_line_lists()

    def _on_right_click(self, event):
        """右键删除最近的线。"""
        if self.image_bgr is None:
            return

        h_idx, h_dist = self._find_nearest_line(event, 'h')
        v_idx, v_dist = self._find_nearest_line(event, 'v')

        if h_idx is None and v_idx is None:
            return

        if h_dist <= v_dist:
            val = self.h_lines[h_idx]
            self.h_lines.pop(h_idx)
        else:
            val = self.v_lines[v_idx]
            self.v_lines.pop(v_idx)

        self.selected_line = None
        self.dragging_line = None
        self._update_display()
        self._update_line_lists()
        self.status_var.set("Line deleted")

    def _on_add_line_dialog(self):
        if self.image_bgr is None:
            messagebox.showwarning("Tip", "Please open an image first")
            return

        val = simpledialog.askinteger("Add Split Line", "Enter coordinate value:")
        if val is None:
            return

        h_sel = self.h_listbox.curselection()
        v_sel = self.v_listbox.curselection()
        kind = 'v' if (v_sel and not h_sel) else 'h'

        h, w = self.image_bgr.shape[:2]
        if kind == 'h':
            if val < 1 or val >= h:
                messagebox.warning("Out of Range", f"y must be between 1 and {h-1}")
                return
            if val not in self.h_lines:
                self.h_lines.append(val)
                self.h_lines.sort()
            self.selected_line = ('h', val)
        else:
            if val < 1 or val >= w:
                messagebox.warning("Out of Range", f"x must be between 1 and {w-1}")
                return
            if val not in self.v_lines:
                self.v_lines.append(val)
                self.v_lines.sort()
            self.selected_line = ('v', val)

        self._update_display()
        self._update_line_lists()

    def _on_delete_selected(self):
        """通过列表删除：按值查找，不依赖索引。"""
        h_sel = self.h_listbox.curselection()
        v_sel = self.v_listbox.curselection()
        if h_sel:
            val_str = self.h_listbox.get(h_sel[0])
            val = int(val_str.split('=')[1])
            if val in self.h_lines:
                self.h_lines.remove(val)
        elif v_sel:
            val_str = self.v_listbox.get(v_sel[0])
            val = int(val_str.split('=')[1])
            if val in self.v_lines:
                self.v_lines.remove(val)
        else:
            messagebox.showwarning("Tip", "Select a line in the list first")
            return
        self.selected_line = None
        self._update_display()
        self._update_line_lists()

    def _on_clear_all(self):
        self.h_lines = []
        self.v_lines = []
        self.selected_line = None
        self.dragging_line = None
        self._update_display()
        self._update_line_lists()
        self.status_var.set("All lines cleared")

    # ------------------------------------------------------------------
    # 检测
    # ------------------------------------------------------------------
    def _on_detect(self):
        if self.image_bgr is None:
            messagebox.showwarning("Tip", "Please open an image first")
            return
        if self.detector is None:
            messagebox.showwarning("Tip", "Image still being analyzed, please wait")
            return
        if self.mode_var.get() == "auto":
            self._run_auto_detect()
        else:
            self._run_grid_detect()

    def _run_auto_detect(self):
        sens = self.sens_var.get()
        max_h = max(0, int(self.max_h_var.get()))
        max_v = max(0, int(self.max_v_var.get()))
        self.status_var.set("Detecting split lines...")

        def worker():
            try:
                h_lines = self.detector.detect_horizontal(sens, max_h)
                v_lines = self.detector.detect_vertical(sens, max_v)
                self.root.after(0, lambda: self._apply_detect_result(
                    h_lines, v_lines))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Detection failed:\n{e}"))
                self.root.after(0,
                    lambda: self.status_var.set("Detection failed"))

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
        self.selected_line = None
        self.dragging_line = None
        self.root.update_idletasks()
        self._update_display()
        self._update_line_lists()
        self.status_var.set(
            f"Done: {len(self.h_lines)} H lines, {len(self.v_lines)} V lines")

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    def _on_export(self):
        if self.image_bgr is None:
            messagebox.showwarning("Tip", "Please open an image first")
            return
        if self.detector is None:
            messagebox.showwarning("Tip", "Image still being analyzed")
            return

        if not self.h_lines and not self.v_lines:
            if not messagebox.askyesno("Tip",
                                       "No split lines. Export entire image?"):
                return

        out_dir = filedialog.askdirectory(title="Select Export Directory")
        if not out_dir:
            return

        try:
            pieces = self.detector.split(self.h_lines, self.v_lines)
            prefix = self.image_name or "split"
            for i, piece in enumerate(pieces, start=1):
                fpath = os.path.join(out_dir, f"{prefix}_{i:02d}.png")
                ok, buf = cv2.imencode(".png", piece)
                if ok:
                    buf.tofile(fpath)
            messagebox.showinfo("Success",
                                f"Exported {len(pieces)} pieces to:\n{out_dir}")
            self.status_var.set(f"Exported {len(pieces)} pieces")
        except Exception as e:
            messagebox.showerror("Export Failed", str(e))

    def _on_save_marked(self):
        if self.image_bgr is None:
            messagebox.showwarning("Tip", "Please open an image first")
            return

        path = filedialog.asksaveasfilename(
            title="Save Marked Image", defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")])
        if not path:
            return

        marked = self.image_bgr.copy()
        for y in self.h_lines:
            cv2.line(marked, (0, y), (marked.shape[1], y), (0, 59, 255), 3)
        for x in self.v_lines:
            cv2.line(marked, (x, 0), (x, marked.shape[0]), (255, 122, 0), 3)

        try:
            ext = os.path.splitext(path)[1] or ".png"
            ok, buf = cv2.imencode(ext, marked)
            if ok:
                buf.tofile(path)
                messagebox.showinfo("Success", f"Saved to:\n{path}")
                self.status_var.set("Marked image saved")
            else:
                messagebox.showerror("Failed", "Encoding failed")
        except Exception as e:
            messagebox.showerror("Save Failed", str(e))


def main():
    root = tk.Tk()
    PuzzleSplitterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
