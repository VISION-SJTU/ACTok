import matplotlib.pyplot as plt
import numpy as np

# 数据准备 (基于2026年基准数据)
models = ['Qwen3-VL-7B', 'Qwen3-VL-32B', 'Llama 3.2-11B', 'Qwen2.5-VL-7B']
raw_edge_data = [11.5, 2.8, 6.5, 10.0]
optimized_edge_data = [45.0, 20.5, 26.3, 225.6]

server_unoptimized = [115.0, 42.0, 85.0, 132.4]
edge_unoptimized = [11.5, 2.8, 6.5, 10.0]

x = np.arange(len(models))
width = 0.38
FONT_SIZE = 32
LABEL_SIZE = 22

def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 10), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=LABEL_SIZE)

# 图 1: Edge - Raw vs Optimized (端侧：原生 vs 优化)
fig1, ax1 = plt.subplots(figsize=(18, 12))
rects1_1 = ax1.bar(x - width/2, raw_edge_data, width, label='Edge - Raw (Unoptimized)', 
                  color='#aec7e8', hatch='//', edgecolor='black', linewidth=2.5)
rects1_2 = ax1.bar(x + width/2, optimized_edge_data, width, label='Edge - Optimized (W4A16)', 
                  color='#1f77b4', hatch='..', edgecolor='black', linewidth=2.5)

ax1.set_ylabel('Inference Speed (Tokens/sec)', fontsize=FONT_SIZE, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=FONT_SIZE, rotation=10)
ax1.tick_params(axis='y', labelsize=FONT_SIZE)

# 提高 Y 轴上限以确保图例不遮挡最高的数据标签 (225.6)
ax1.set_ylim(0, 320)
ax1.legend(fontsize=FONT_SIZE, loc='upper left', frameon=True, shadow=True)

autolabel(rects1_1, ax1)
autolabel(rects1_2, ax1)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('edge_comparison_v28.png')
plt.close()

# 图 2: Server vs Edge - Unoptimized (云端 vs 边缘：原生对比)
fig2, ax2 = plt.subplots(figsize=(18, 12))
rects2_1 = ax2.bar(x - width/2, server_unoptimized, width, label='Server (A100) - Raw BF16', 
                  color='#1f77b4', hatch='//', edgecolor='black', linewidth=2.5)
rects2_2 = ax2.bar(x + width/2, edge_unoptimized, width, label='Edge (Orin) - Raw FP16', 
                  color='#ff7f0e', hatch='\\\\', edgecolor='black', linewidth=2.5)

ax2.set_ylabel('Inference Speed (Tokens/sec)', fontsize=FONT_SIZE, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=FONT_SIZE, rotation=10)
ax2.tick_params(axis='y', labelsize=FONT_SIZE)

# 提高 Y 轴上限以确保图例不遮挡最高的数据标签 (132.4)
ax2.set_ylim(0, 145)
ax2.legend(fontsize=FONT_SIZE, loc='upper left', frameon=True, shadow=True)

autolabel(rects2_1, ax2)
autolabel(rects2_2, ax2)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('unoptimized_comparison_v28.png')
plt.close()