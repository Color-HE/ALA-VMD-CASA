import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vmdpy import VMD


def perform_vmd_and_save(file_path, K, alpha=2000, tau=0, DC=0, init=1, tol=1e-6):
    # 读取数据
    df = pd.read_csv(file_path,encoding='gbk')
    #df = pd.read_csv(file_path) #二选一，如果在这里报错了
    columns = df.columns
    time_series = df.iloc[:, -1].values  # 最后一列作为分解目标
    other_columns = df.iloc[:, :-1]  # 除了分解列的其他列

    # VMD 分解
    u, u_hat, omega = VMD(time_series, alpha, tau, K, DC, init, tol)

    # 可视化分解结果
    plt.figure(figsize=(10, 6))
    for i in range(K):
        plt.subplot(K, 1, i + 1)
        plt.plot(u[i, :], label=f'IMF {i}')
        plt.legend()
    plt.suptitle("VMD Decomposition")
    plt.savefig('VMD Decomposition.png',dpi=300)
    plt.show()

    # 保存每个模态到单独的 CSV 文件
    for i in range(K):
        imf_df = other_columns.copy()
        imf_df[f'imf{i}'] = u[i, :]
        output_filename = f'vmd_imf{i}.csv'
        imf_df.to_csv(output_filename, index=False)
        print(f"Saved {output_filename}")


# 示例调用
perform_vmd_and_save('merged_interpolated.csv', K=5,alpha=14523) #在0.xx优化参数.py的最后结果要手动写到这里面去
