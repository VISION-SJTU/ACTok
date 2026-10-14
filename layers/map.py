import matplotlib.pyplot as plt
import numpy as np

# 方法与平均误差
methods = [
    "LBP+RR",
    "Crowd CNN",
    "Fine-tuned\nCrowd CNN",
    "Luca Fiaschi et al.",
    "Ke et al.",
    "Crowd CNN+RR"
]
avg_error = [31.0, 14.1, 12.9, 26.7, 16.5, 10.7]

# 绘图
plt.figure(figsize=(9, 5))
bars = plt.bar(methods, avg_error, color=['gray', '#6baed6', '#4292c6', '#9ecae1', '#74c476', '#31a354'])

# 在柱状图上标注数值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{height:.1f}',
             ha='center', va='bottom', fontsize=10)

# 美化样式
plt.title('Average Error Comparison Across Methods', fontsize=14)
plt.ylabel('Average Error (%)', fontsize=12)
plt.xticks(rotation=30, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.ylim(0, 35)

plt.tight_layout()
plt.show()
