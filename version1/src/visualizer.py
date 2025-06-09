import matplotlib.pyplot as plt

def visualize_metrics(metrics):

    # 提取评分维度及得分
    dimensions = list(metrics.keys())
    scores = [metrics[dim]["score"] for dim in dimensions]

    # 固定图像显示的大小
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    ax.bar(dimensions, scores, color='skyblue')
    ax.set_title("Rating Dimensions Bar Char")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 10)  # y轴最大值固定为10
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    return fig
