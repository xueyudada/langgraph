#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distribution of counseling topic categories.
The categories follow the predefined vocabulary (Family, Emotion, Marriage, etc.).
Counts below are illustrative (synthetic) and only meant to reproduce the figure style.
"""

import matplotlib.pyplot as plt
import numpy as np

# ----- Style: serif font to match the paper -----
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Nimbus Roman"]
plt.rcParams["font.size"] = 12

# ----- Synthetic data (turn-level topic counts) -----
# Ordered so that the slice sequence reads cleanly around the circle.
labels = [
    "Family",                 # 保持不变
    "Emotion",                # 保持不变
    "Marriage",               # 保持不变
    "Self-exploration",       # 类似 Self-awareness
    "Interpersonal",          # 类似 Relationship
    "Personal Growth",        # 类似 Growth
    "Behavioral Control",     # 类似 Behavior
    "Career Guidance",        # 类似 Career
    "Mental Therapy",         # 类似 Treatment
    "Psychoeducation",        # 类似 Psychological Knowledge
    "Sexuality",              # 类似 Sex
]
counts = np.array([2218, 1945, 1510, 1236, 942, 718, 533, 427, 218, 116, 37],
                  dtype=float)
pcts = counts / counts.sum() * 100.0

# ----- Soft pastel palette (close to the reference figure) -----
colors = [
    "#cfe0f1",  # light blue
    "#fde9c9",  # light peach
    "#f3c9a8",  # light orange
    "#ececec",  # light gray
    "#f6ddcb",  # peachy
    "#d7ead9",  # mint green
    "#cdb8db",  # soft lavender
    "#bcd2ee",  # periwinkle
    "#ffe39a",  # gold
    "#c7e3a6",  # light green
    "#e8a7a0",  # muted salmon
]

# Slices smaller than this get their percentage placed OUTSIDE (with the name),
# mimicking the "Sex, 0.20%" style of the reference figure.
SMALL = 3.0

# Manual placement for the tiny clustered slices so their leader-labels don't
# overlap. Keyed by label -> (text x, text y, horizontal alignment).
# 这里的键名已对应更新为新的类似词汇
TINY_LABEL_POS = {
    "Mental Therapy":   (0.62, 1.40, "left"),
    "Psychoeducation":  (1.05, 1.18, "left"),
    "Sexuality":        (1.40, 0.92, "left"),
}

fig, ax = plt.subplots(figsize=(8.6, 6.8))

wedges, _ = ax.pie(
    counts,
    colors=colors,
    startangle=90,          # start at the top
    counterclock=False,     # go clockwise, like the reference
    wedgeprops=dict(edgecolor="white", linewidth=1.3),
)

# ----- Manual labelling: name outside, percentage inside (or outside if tiny) -----
# Optional curated positions for a few large-slice NAMES
# 这里的键名已更新为 "Family"
BIG_LABEL_POS = {
    "Family": (1.20, 0.55, "left"),
}

for w, lab, p in zip(wedges, labels, pcts):
    ang = np.deg2rad((w.theta1 + w.theta2) / 2.0)
    x, y = np.cos(ang), np.sin(ang)
    ha = "left" if x >= 0 else "right"

    if p >= SMALL:
        # percentage inside the slice
        ax.text(0.62 * x, 0.62 * y, f"{p:.2f}%",
                ha="center", va="center", fontsize=11.5, color="black")
        # category name just outside the slice (curated if needed)
        if lab in BIG_LABEL_POS:
            tx, ty, tha = BIG_LABEL_POS[lab]
            ax.text(tx, ty, lab, ha=tha, va="center", fontsize=12)
        else:
            ax.annotate(lab, xy=(x, y), xytext=(1.18 * x, 1.18 * y),
                        ha=ha, va="center", fontsize=12,
                        annotation_clip=False)
    else:
        # tiny slice: name + percentage together, outside, with a leader line.
        # Use curated text positions to avoid the labels overlapping.
        tx, ty, tha = TINY_LABEL_POS[lab]
        ax.annotate(f"{lab}, {p:.2f}%",
                    xy=(x, y), xytext=(tx, ty),
                    ha=tha, va="center", fontsize=10.5,
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color="gray",
                                    lw=0.8, connectionstyle="arc3,rad=0"))

ax.set_aspect("equal")
ax.set_xlim(-1.8, 1.9)
ax.set_ylim(-1.5, 1.6)

# ----- Caption below the figure (like the reference) -----
# fig.text(0.5, 0.045,
#          "Figure X: Distribution of counseling topics.",
#          ha="center", fontsize=13)

plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.12)
fig.savefig("technique_distribution.png", dpi=300, bbox_inches="tight")
print("saved technique_distribution.png")