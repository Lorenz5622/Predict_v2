import matplotlib.pyplot as plt
import numpy as np

# x 轴
k = np.arange(1, 9)

# 这里填你的数据
gsm8k = [45.2, 62.1, 63.9, 63.0, 62.3, 59.1, 57.2, 57.5]
strategyqa = [64.8, 72.8, 73.4, 74.3, 73.5, 73.2, 72.3, 72.2]
mbpp = [38.6, 46.1, 44.2, 46.1, 44.8, 44.0, 43.8, 42.4]
humaneval = [26.4, 33.6, 34.1, 37.0, 34.1, 30.0, 29.4, 31.6]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 18,   # y 轴标题大小
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
plt.subplots_adjust(wspace=0.4, bottom=0.28)

plots = [
    {
        "data": gsm8k,
        "title": "(a) GSM8K",
        "ylabel": "Accuracy",
        "ylim": (45, 69),
        "yticks": [45, 51, 57, 63, 69],
    },
    {
        "data": strategyqa,
        "title": "(b) StrategyQA",
        "ylabel": "",
        "ylim": (62, 78),
        "yticks": [62, 66, 70, 74, 78],
    },
    {
        "data": mbpp,
        "title": "(c) MBPP",
        "ylabel": "Pass@1",
        "ylim": (38, 50),
        "yticks": [38, 41, 44, 47, 50],
    },
    {
        "data": humaneval,
        "title": "(d) HumanEval",
        "ylabel": "",
        "ylim": (25, 43),
        "yticks": [25.0, 29.5, 34.0, 38.5, 43.0],
    },
]

for ax, cfg in zip(axes, plots):
    ax.plot(
        k,
        cfg["data"],
        color="blue",
        marker="o",
        markersize=7,
        linewidth=1.8,
        markerfacecolor="white",
        markeredgewidth=1.5,
    )

    ax.set_xlim(0.7, 8.3)
    ax.set_xticks(k)
    ax.set_xlabel("k", fontsize=14, labelpad=4)   # 调小 k 的字号，并略微留空
    ax.set_ylim(*cfg["ylim"])
    ax.set_yticks(cfg["yticks"])

    if cfg["ylabel"]:
        ax.set_ylabel(cfg["ylabel"])

    ax.grid(True, color="#999999", linewidth=0.7, alpha=0.7)
    ax.set_title(cfg["title"], y=-0.5, fontsize=20)  # 标题再往下移一点

# 保存图片
plt.savefig("output_plot.png", dpi=300, bbox_inches="tight")
plt.close()

print("图片已保存为 output_plot.png")