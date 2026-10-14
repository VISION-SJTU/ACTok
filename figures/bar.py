import matplotlib.pyplot as plt
import numpy as np

# 数据
x_data = ["ImageNet", "Fine-grained", "Segmentation", "Depth estimation"]
data1 = [83.6, 86.2, 22.2, 0.347]  # MAE
data2 = [85.6, 91.3, 51.0, 0.300]  # UMT

labels = ["MAE (ViT-B)", "UMT (ViT-B, ours)"]

x = np.arange(len(x_data))  # x位置
width = 0.4 # 柱子宽度

fig, ax1 = plt.subplots(figsize=(5, 5))
ax2 = ax1.twinx()  # 第二个y轴

# 柱状图
rects1_main = ax1.bar(x[:-1] - width/2, data1[:-1], width, label=labels[0], color="#5470C6")
rects2_main = ax1.bar(x[:-1] + width/2, data2[:-1], width, label=labels[1], color="#92CC74")

rects1_depth = ax2.bar(x[-1] - width/2, data1[-1], width, color="#5470C6")
rects2_depth = ax2.bar(x[-1] + width/2, data2[-1], width, color="#92CC74")

# 数值标注函数
def autolabel(rects, ax, fmt="{:.1f}"):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                fmt.format(height),
                ha='center', va='bottom', fontsize=9)

# 标注数值
autolabel(rects1_main, ax1, "{:.1f}")
autolabel(rects2_main, ax1, "{:.1f}")
autolabel(rects1_depth, ax2, "{:.3f}")
autolabel(rects2_depth, ax2, "{:.3f}")

# x轴
# x轴
ax1.set_xticks(x)
ax1.set_xticklabels(x_data, rotation=20, ha="right") 

# y轴标签
ax1.set_ylabel("Accuracy / mIoU (%)")
ax2.set_ylabel("RMSE (↓)")

# 图例放在图上方，间隔更大
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)

# plt.title("Comparison of MAE vs UMT (ViT-B)")
plt.tight_layout()
# plt.show()
plt.savefig("bar.png", dpi=500)
