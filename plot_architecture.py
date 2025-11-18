import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Patch
import numpy as np
colors = {
    "input":  "#ffa7ca",
    "down":   "#8ecaff",
    "res":    "#ffcc88",
    "bottle": "#e8a2ff",
    "attn":   "#a4f3c1",
    "up":     "#8ecaff",
    "output": "#a4f3c1",
    "time_emb": "#ff6666",
}

# name, H, W, Cout, res_label, ch_label, color
stages = [
    ("Input",        28, 28,   3, "28×28", "3",                        colors["input"]),
    ("Init Conv",    28, 28,  64, "28×28", "3 → 64",                   colors["down"]),
    ("Encoder 1",    28, 28,  64, "28×28", "64 → 64",                  colors["res"]),
    ("Downsample 1", 14, 14, 128, "14×14", "64 → 128",                 colors["down"]),
    ("Encoder 2",    14, 14, 128, "14×14", "128 → 128",                colors["res"]),
    ("Downsample 2",  7,  7, 256,  "7×7",  "128 → 256",                colors["down"]),
    ("Bottleneck",    7,  7, 256,  "7×7",  "256 → 256",                colors["bottle"]),
    ("Upsample 2",   14, 14, 128, "14×14", "256 → 128 (+concat→256)",  colors["up"]),
    ("Decoder 2",    14, 14, 128, "14×14", "256 → 128",                colors["res"]),
    ("Upsample 1",   28, 28,  64, "28×28", "128 → 64 (+concat→128)",   colors["up"]),
    ("Decoder 1",    28, 28,  64, "28×28", "128 → 64",                 colors["res"]),
    ("Output",       28, 28,   3, "28×28", "64 → 3",                   colors["output"]),
]

# 最大解析度，拿來做高度 normalize
H_max = max(H for (_, H, _, _, _, _, _) in stages)
C_min = min(C for (_, _, _, C, _, _, _) in stages)
C_max = max(C for (_, _, _, C, _, _, _) in stages)


fig, ax = plt.subplots(figsize=(22,5), dpi=300)
ax.axis("off")

x_start = 0.5
y_base  = 0.8       # 所有 block 的下緣 baseline
h_base  = 3         # 最大高度（對應 H_max）
w_min = 0.05        # 最窄的 block 寬度
w_max = 0.5         # 最寬的 block 寬度 
dx      = 0.2       # block 間距
positions = []
# ------- 畫每一個 block -------
heightest = 0
x = x_start
for name, H, W, Cout, res_label, ch_label, color in stages:
    # 高度依 H 比例縮放
    h = h_base * (H / H_max)
    norm = (np.log(Cout) - np.log(C_min)) / (np.log(C_max) - np.log(C_min))
    w = w_min + (w_max - w_min) * norm  # 寬度也依 Cout 比例縮放
    heightest = h if h>heightest else heightest

    rect = Rectangle((x, y_base), w, h, facecolor=color, edgecolor="black")
    ax.add_patch(rect)

    # Stage 名稱（上方）
    ax.text(x + w/2, y_base + heightest*1.1, name,
            ha="center", va="center", fontsize=15, fontweight="bold")

    # block 底下顯示 matrix size (H×W×Cout)
    ax.text(x + w/2, y_base - 0.25,
            f"{H}×{W}×{Cout}",
            ha="center", va="center", fontsize=15)

    positions.append((x, y_base, h, w))
    x += w + dx

# ------- 畫箭頭（水平資料流）-------
for i in range(len(positions) - 1):
    x0, y0, h0, w0 = positions[i]
    x1, y1, h1, w1 = positions[i+1]
    arrow = FancyArrowPatch(
        (x0 + w0, y0 + h0/2),
        (x1,     y1 + h1/2),
        arrowstyle="-|>",
        mutation_scale=10,
        lw=1
    )
    ax.add_patch(arrow)

# -------- time embedding arrows (紅色) --------

first_idx = 2
last_idx  = len(positions) - 2

# 統一定義 time embedding 線的 y 座標
y_time = positions[first_idx][1] - 0.7   # = y_base - 0.7

x_left  = positions[first_idx][0] + positions[first_idx][3]/2
x_right = positions[last_idx][0] + positions[last_idx][3]/2

# 1) 水平線
ax.plot([x_left, x_right], [y_time, y_time],
        color=colors["time_emb"], lw=1.5)

ax.text((x_left + x_right)/2, y_time - 0.25,
        "time embedding",
        ha="center", va="top",
        fontsize=15, color=colors["time_emb"],fontweight="bold")


for i, (name, H, W, Cout, res_label, ch_label, color) in enumerate(stages):
    if any(k in name for k in ("Encoder", "Decoder", "Bottleneck")):
        x0, y0, h0, w0 = positions[i]

        arrow_start = y0 - 0.7               # block 的底部
        arrow_end   = y0 - 0.35         # 往下降一段（可調）

        ta = FancyArrowPatch(
            (x0 + w0/2, arrow_start),
            (x0 + w0/2, arrow_end),
            arrowstyle="-|>",
            mutation_scale=12,
            lw=1.6,
            color=colors["time_emb"],
        )
        ax.add_patch(ta)

# ------- 調整座標範圍與標題 -------
right_edge = positions[-1][0] + positions[-1][3]

ax.set_xlim(0, right_edge + 0.5)
ax.set_ylim(0, heightest * 1.6)

ax.text((x_start + right_edge) / 2,
        heightest * 1.5,
        "DDPM U-Net Architecture",
        ha="center", fontsize=15, fontweight="bold")

# legend（可保留或拿掉）
legend_patches = [
    Patch(facecolor=colors["down"],   edgecolor="black", label="Down / Up Conv"),
    Patch(facecolor=colors["res"],    edgecolor="black", label="Residual Block"),
    Patch(facecolor=colors["bottle"], edgecolor="black", label="Bottleneck"),
    Patch(facecolor=colors["input"],  edgecolor="black", label="Input"),
    Patch(facecolor=colors["output"], edgecolor="black", label="Output"),
    # Patch(facecolor=colors["time_emb"], edgecolor="black", label="Time Embedding"),
]
ax.legend(handles=legend_patches,
          loc="lower center", bbox_to_anchor=(0.5, -0.35),
          ncol=10, fontsize=15, frameon=False)

plt.tight_layout()
plt.savefig("./report_figures/unet_scaled_height.png")
print("Saved to: ./report_figures/unet_scaled_height.png")