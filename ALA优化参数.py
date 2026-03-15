import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import hilbert
from scipy.io import savemat
from vmdpy import VMD
import time
import numpy as np
import scipy.special as sp


# ==================== 1. ALA 优化器 ====================
def ALA(N, Max_iter, lb, ub, dim, fobj):
    """
    Artificial Lemming Algorithm (ALA) for optimization.
    """
    tic = time.time()

    # Initialize population
    X = initialization(N, dim, ub, lb)

    # Initialize best solution
    Position = np.zeros(dim)
    Score = float('inf')
    fitness = np.zeros(N)
    Convergence_curve = []
    vec_flag = np.array([1, -1])  # Directional flag

    # Evaluate initial population
    for i in range(N):
        fitness[i] = fobj(X[i, :])
        if fitness[i] < Score:
            Position = X[i, :].copy()
            Score = fitness[i]

    Iter = 1
    Convergence_curve.append(Score)

    while Iter <= Max_iter:
        RB = np.random.randn(N, dim)  # Brownian motion
        F = vec_flag[np.random.choice([0, 1])]  # Random directional flag

        theta = 2 * np.arctan(1 - Iter / Max_iter)  # Time-varying parameter

        Xnew = np.copy(X)  # New population

        for i in range(N):
            E = 2 * np.log(1 / np.random.rand()) * theta

            if E > 1:
                if np.random.rand() < 0.3:
                    r1 = 2 * np.random.rand(dim) - 1
                    Xnew[i, :] = Position + F * RB[i, :] * (
                                r1 * (Position - X[i, :]) + (1 - r1) * (X[i, :] - X[np.random.randint(0, N), :]))
                else:
                    r2 = np.random.rand() * (1 + np.sin(0.5 * Iter))
                    Xnew[i, :] = X[i, :] + F * r2 * (Position - X[np.random.randint(0, N), :])
            else:
                if np.random.rand() < 0.5:
                    radius = np.sqrt(np.sum((Position - X[i, :]) ** 2))
                    r3 = np.random.rand()
                    spiral = radius * (np.sin(2 * np.pi * r3) + np.cos(2 * np.pi * r3))
                    Xnew[i, :] = Position + F * X[i, :] * spiral * np.random.rand()
                else:
                    G = 2 * np.sign(np.random.rand() - 0.5) * (1 - Iter / Max_iter)
                    Xnew[i, :] = Position + F * G * levy(dim) * (Position - X[i, :])

        # Boundary handling
        for i in range(N):
            Flag4ub = Xnew[i, :] > ub
            Flag4lb = Xnew[i, :] < lb
            Xnew[i, :] = Xnew[i, :] * (~Flag4ub & ~Flag4lb) + ub * Flag4ub + lb * Flag4lb

            newPopfit = fobj(Xnew[i, :])
            if newPopfit < fitness[i]:
                X[i, :] = Xnew[i, :].copy()
                fitness[i] = newPopfit

                if fitness[i] < Score:
                    Position = X[i, :].copy()
                    Score = fitness[i]

        Convergence_curve.append(Score)
        Iter += 1

    toc = time.time()
    print(f"Time taken: {toc - tic:.4f} seconds")

    return Score, Position, Convergence_curve


def initialization(N, dim, ub, lb):
    Boundary = len(ub)
    new_lb = lb
    new_ub = ub

    if Boundary == 1:
        X = np.random.rand(N, dim) * (ub - lb) + lb
        new_lb = lb * np.ones(dim)
        new_ub = ub * np.ones(dim)
    else:
        X = np.zeros((N, dim))
        for i in range(dim):
            ubi = ub[i]
            lbi = lb[i]
            X[:, i] = np.random.rand(N) * (ubi - lbi) + lbi

    return X


def levy(d):
    beta = 1.5
    sigma = (sp.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
                sp.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(1, d) * sigma
    v = np.random.randn(1, d)
    step = u / np.abs(v) ** (1 / beta)
    return step[0]


# ==================== 2. VMD 分解函数 ====================
def vmd_decompose(signal, alpha, tau, K, DC, init, tol):
    imfs, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)
    df_vmd = pd.DataFrame(imfs.T)
    df_vmd.columns = [f'imf{i + 1}' for i in range(K)]
    return df_vmd


# ==================== 3. 目标函数：最小化 IMF 的样本熵 ====================
def sample_entropy(u, m=2):
    """Compute sample entropy of signal u"""
    n = len(u)
    if n < m + 1:
        return np.nan
    E = []
    for j in range(n - m):
        s = np.sum(u[j:j + m])
        R = np.sum(np.abs(u[j + m:j + (2 * m)] - u[j:j + m]))
        S = np.log(R / s)
        E.append(S)
    return np.mean(E) if E else np.nan


def training(X):
    """
    Objective function to minimize: average sample entropy of IMFs.
    X = [alpha, K] -> K must be int
    """
    if len(X) == 2:
        alpha = int(X[0])
        K = int(X[1])

        # VMD decomposition
        try:
            imfs, _, _ = VMD(data["brent"], alpha, tau, K, DC, init, tol)
        except Exception as e:
            print(f"VMD failed with alpha={alpha}, K={K}: {e}")
            return 1e6  # large penalty

        m = 2  # Sample entropy parameter (embedding dimension)
        EP = []
        for i in range(K):
            H = np.abs(hilbert(imfs[i, :]))  # Analytic signal magnitude
            if len(H) > 2 * m:
                E = sample_entropy(H, m=m)
                EP.append(E)
            else:
                EP.append(1e6)

        s = np.mean(EP) if EP else 1e6
        return s
    else:
        return 1e6


# ==================== 4. 主程序：使用 ALA 优化 VMD 参数 ====================
if __name__ == "__main__":
    # Load data
    data = pd.read_csv('brent单独分解数据集.csv',encoding='gbk')  # 替换为你的数据文件路径
    signal = data['brent'].to_numpy()

    # VMD parameters
    tau = 0
    DC = 0
    init = 1
    tol = 1e-7

    # ALA parameters
    N = 10  # Population size
    Max_iter = 50  # Max iterations
    dim = 2  # Two variables: alpha, K
    lb = [500, 5]  # Lower bounds: alpha_min, K_min
    ub = [100000, 12]  # Upper bounds: alpha_max, K_max

    # Run ALA
    best_score, best_pos, curve = ALA(N, Max_iter, lb, ub, dim, training)

    # Extract best parameters
    alpha_opt = int(best_pos[0])
    K_opt = int(best_pos[1])

    print(f"Best Alpha: {alpha_opt}")
    print(f"Best K (IMFs): {K_opt}")

    # Save results
    savemat('ALA_VMD_optimization.mat', {
        'best_alpha': alpha_opt,
        'best_K': K_opt,
        'best_score': best_score,
        'convergence_curve': curve
    })

    # Plot convergence curve
    plt.figure(figsize=(10, 5))
    plt.plot(curve)
    plt.title("Convergence Curve - ALA Optimization")
    plt.xlabel("Iteration")
    plt.ylabel("Average Sample Entropy")
    plt.grid(True)
    plt.savefig('Convergence Curve - ALA Optimization.png',dpi=300)
    plt.show()

    # Re-run VMD with optimal parameters
    df_vmd = vmd_decompose(data["brent"], alpha_opt, tau, K_opt, DC, init, tol)
    print("VMD Decomposition Complete with optimal parameters.")

    # Plot IMFs
    fig, axes = plt.subplots(nrows=K_opt, ncols=1, figsize=(10, 12))
    color_cycle = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'] * K_opt)
    for i, col in enumerate(df_vmd.columns):
        df_vmd[col].plot(ax=axes[i], color=next(color_cycle))
        axes[i].set_title(f"IMF {i + 1}")
        axes[i].tick_params(axis='y', labelsize=10)
    plt.suptitle('VMD Decomposition Results (Optimal Parameters)', fontsize=14, y=0.93)
    plt.tight_layout()
    plt.savefig('VMD Decomposition Results (Optimal Parameters).png',dpi=300)
    plt.show()

    # Plot frequency components
    fig, axes = plt.subplots(nrows=K_opt, ncols=1, figsize=(10, 12))
    for i, col in enumerate(df_vmd.columns):
        freq = np.fft.fftfreq(len(df_vmd[col]), d=1)
        magnitude = np.abs(np.fft.fft(df_vmd[col]))
        axes[i].plot(freq[:len(freq) // 2], magnitude[:len(freq) // 2])
        axes[i].set_title(f"IMF {i + 1} Frequency Spectrum")
    plt.tight_layout()
    plt.show()
