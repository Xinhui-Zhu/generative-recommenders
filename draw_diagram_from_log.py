import re
import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.DataFrame()
# 文件路径
for log_file_path in os.listdir("log"):
    log_file_path = os.path.join("log", log_file_path)
    # 定义正则表达式提取 eval @ epoch 的行
    pattern = r"INFO:root:rank 0: eval @ epoch (\d+) .*?NDCG@10 (\d+\.\d+), NDCG@50 (\d+\.\d+), HR@10 (\d+\.\d+), HR@50 (\d+\.\d+), MRR (\d+\.\d+)"

    # 读取 log 文件
    with open(log_file_path, "r") as file:
        log_data = file.read()

    # 匹配数据
    matches = re.findall(pattern, log_data)

    # Create a DataFrame
    data = pd.DataFrame(matches, columns=["epoch", "NDCG@10", "NDCG@50", "HR@10", "HR@50", "MRR"])
    data['log_name'] = log_file_path
    data = data.astype({"epoch": int, "NDCG@10": float, "NDCG@50": float, "HR@10": float, "HR@50": float, "MRR": float})
    df = pd.concat([df,data])

# Plotting
metrics = ["NDCG@10", "NDCG@50", "HR@10", "HR@50", "MRR"]
for metric in metrics:
    plt.figure()
    for log_name in df.log_name.unique():
        data = df[df["log_name"]==log_name]
        plt.plot(data["epoch"], data[metric], marker='.', label=log_name)
        plt.title(f"{metric} over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.grid()
        plt.legend()
        plt.show()
        # plt.savefig(f"diagram/{log_file_path.split('/')[-1].split('.')[0]}-{metric}.png")
        plt.savefig(f"diagram/{metric}.png")
