import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

plt.rc('font', family='Arial')
plt.style.use("ggplot")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from models import CASA
from utils.timefeatures import time_features


# =========================
# 1) VMD（纯 numpy 实现）
# =========================
def VMD(signal, alpha, tau, K, DC=0, init=1, tol=1e-7, N_iter=500):
    f = np.asarray(signal, dtype=float)
    N = len(f)

    f_mirror = np.concatenate([f[:N//2][::-1], f, f[-N//2:][::-1]])
    T = len(f_mirror)

    freqs = np.fft.rfftfreq(T, d=1.0)
    f_hat = np.fft.rfft(f_mirror)

    u_hat = np.zeros((K, len(freqs)), dtype=np.complex128)
    omega = np.zeros(K, dtype=float)
    lam = np.zeros(len(freqs), dtype=np.complex128)

    if init == 1:
        omega = 0.5 * np.arange(K) / K
    elif init == 2:
        omega = np.sort(np.random.rand(K) * 0.5)
    else:
        omega = np.zeros(K)

    if DC:
        omega[0] = 0.0

    uDiff = tol + 1.0
    n = 0

    while (uDiff > tol) and (n < N_iter):
        u_hat_prev = u_hat.copy()

        for k in range(K):
            sum_others = np.sum(u_hat, axis=0) - u_hat[k]
            residual = f_hat - sum_others - lam / 2.0

            denom = 1.0 + 2.0 * alpha * (freqs - omega[k])**2
            u_hat[k] = residual / denom

            if not (DC and k == 0):
                power = np.abs(u_hat[k])**2
                omega[k] = np.sum(freqs * power) / (np.sum(power) + 1e-12)

        lam = lam + tau * (np.sum(u_hat, axis=0) - f_hat)
        uDiff = np.mean([np.linalg.norm(u_hat[k] - u_hat_prev[k])**2 for k in range(K)])
        n += 1

    u = np.zeros((K, T), dtype=float)
    for k in range(K):
        u[k] = np.fft.irfft(u_hat[k], n=T)

    u = u[:, N//2: N//2 + N]
    return u


# =========================
# 2) 不泄露的切窗：label_len 与 encoder 尾部 overlap，并返回预测点的全局索引
# =========================
def build_loader_univariate_with_index(
    window, pred_len, batch_size,
    data, data_mark,
    global_start_index,
    shuffle
):
    """
    data: (M,1) 对某个 IMF 的序列（某段：train/fit/test）
    data_mark: (M,dm)
    global_start_index: 该段 data 在原始全局序列中的起始索引（用于对齐 brent 真值）
    返回：
      loader: 每个样本 (x, y, x_mark, y_mark, idx)
      idx: 预测点在全局序列中的 index（对应 pred_len 最后一步）
    """
    seq_len = window
    label_len = int(window / 2)

    data = np.asarray(data)
    data_mark = np.asarray(data_mark)

    M = len(data)
    num = M - (seq_len + pred_len) + 1
    if num <= 0:
        raise ValueError("样本长度不足，请减小 window/pred_len 或增加数据量。")

    xs, ys, xms, yms, idxs = [], [], [], [], []

    for i in range(num):
        # encoder 输入
        x_seq = data[i: i + seq_len]  # (seq_len,1)
        x_mark = data_mark[i: i + seq_len]

        # decoder 的 y_seq：从 encoder 末尾往回 label_len（overlap），再接 pred_len
        y_start = i + seq_len - label_len
        y_end = i + seq_len + pred_len
        y_seq = data[y_start: y_end]  # (label_len+pred_len,1)
        y_mark = data_mark[y_start: y_end]

        # 预测点（取 pred_len 最后一步）在全局的索引
        pred_global_idx = global_start_index + (i + seq_len + pred_len - 1)

        xs.append(x_seq)
        ys.append(y_seq)
        xms.append(x_mark)
        yms.append(y_mark)
        idxs.append(pred_global_idx)

    x_temp = torch.tensor(np.array(xs), dtype=torch.float32)
    y_temp = torch.tensor(np.array(ys), dtype=torch.float32)
    x_temp_mark = torch.tensor(np.array(xms), dtype=torch.float32)
    y_temp_mark = torch.tensor(np.array(yms), dtype=torch.float32)
    idx_temp = torch.tensor(np.array(idxs), dtype=torch.long)

    ds = TensorDataset(x_temp, y_temp, x_temp_mark, y_temp_mark, idx_temp)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return loader


def make_decoder_input(y, label_len, pred_len, device):
    dec_inp = torch.zeros_like(y).to(device)
    dec_inp[:, :label_len, :] = y[:, :label_len, :].to(device)
    return dec_inp


# =========================
# 3) 训练/预测/评估
# =========================
def model_train(net, train_loader, pred_len, optimizer, criterion, num_epochs, device, print_train=False):
    train_loss = []
    print_frequency = max(1, num_epochs // 20)

    for epoch in range(num_epochs):
        net.train()
        total = 0.0

        for x, y, x_mark, y_mark, _idx in train_loader:
            x = x.to(device)
            y = y.to(device)
            x_mark = x_mark.to(device)
            y_mark = y_mark.to(device)

            optimizer.zero_grad()
            preds = net(x, x_mark, y, y_mark, None)  # teacher forcing

            preds = preds[:, -pred_len:, :].squeeze()
            y_true = y[:, -pred_len:, :].squeeze()

            loss = criterion(preds, y_true)
            loss.backward()
            optimizer.step()
            total += loss.item()

        avg = total / len(train_loader)
        train_loss.append(avg)

        if print_train and ((epoch + 1) % print_frequency == 0 or (epoch + 1) == num_epochs):
            print(f"Epoch: {epoch + 1}, Train Loss: {avg:.6f}")

    return net, train_loss


@torch.no_grad()
def model_predict(net, x, x_mark, y, y_mark, label_len, pred_len, device):
    net.eval()
    x = x.to(device)
    x_mark = x_mark.to(device)
    y = y.to(device)
    y_mark = y_mark.to(device)

    dec_inp = make_decoder_input(y, label_len, pred_len, device)
    pred = net(x, x_mark, dec_inp, y_mark, None)
    pred = pred[:, -pred_len:, :].detach().cpu().numpy()
    true = y[:, -pred_len:, :].detach().cpu().numpy()
    return true, pred


def cal_eval(y_real, y_pred):
    y_real = np.array(y_real).ravel()
    y_pred = np.array(y_pred).ravel()

    r2 = r2_score(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_real, y_pred)
    mape = np.mean(np.abs((y_real - y_pred) / (y_real + 1e-8))) * 100

    return pd.DataFrame({'R2': [r2], 'MSE': [mse], 'RMSE': [rmse], 'MAE': [mae], 'MAPE': [mape]})


# =========================
# 4) CASA 配置
# =========================
class Config:
    def __init__(self, window, pred_len, batch_size, num_epochs, lr, data_dim):
        self.seq_len = window
        self.label_len = int(window / 2)
        self.pred_len = pred_len
        self.freq = 'b'

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = lr
        self.stop_ratio = 0.2

        self.dec_in = data_dim
        self.enc_in = data_dim
        self.c_out = 1

        self.d_model = 64
        self.n_heads = 8
        self.dropout = 0.1
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 64
        self.factor = 5
        self.activation = 'gelu'
        self.channel_independence = 0

        self.patch_len = 16
        self.stride = 8
        self.top_k = 5
        self.num_kernels = 6

        self.embed = 'timeF'
        self.output_attention = 0
        self.distil = 1
        self.task_name = 'short_term_forecast'

        self.kernel = 3


# =========================
# 5) 主流程：逐 IMF 预测并相加，与真实 brent 比较
# =========================
def main():
    CSV_PATH = r"C:\改进itransformer\VMD-CASA\data\brent单独分解数据集.csv"
    TARGET_COL = "brent"
    DATE_COL = "date"

    train_ratio = 0.6
    val_ratio = 0.8

    # CASA 超参
    window = 10
    pred_len = 1
    batch_size = 64
    num_epochs = 20
    learning_rate = 0.001

    # VMD 超参（你也可以接回你的 optuna 优化逻辑）
    K = 3
    alpha = 500
    tau = 0.0
    DC = 0

    freq = "B"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("CUDA available:", torch.cuda.is_available())
    print("device:", device)

    df = pd.read_csv(CSV_PATH, encoding="gbk")
    if DATE_COL not in df.columns or TARGET_COL not in df.columns:
        raise ValueError(f"数据必须包含列: {DATE_COL}, {TARGET_COL}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    series = df[TARGET_COL].values.astype(float)  # (N,)
    N = len(series)

    # VMD 分解
    imfs = VMD(series, alpha=alpha, tau=tau, K=K, DC=DC, init=1, tol=1e-7, N_iter=1000)  # (K,N)

    # 时间特征
    df_stamp = df[[DATE_COL]].copy()
    df_stamp[DATE_COL] = pd.to_datetime(df_stamp[DATE_COL])
    data_stamp = time_features(df_stamp, timeenc=1, freq=freq)

    train_size = int(N * train_ratio)
    val_size = int(N * val_ratio)
    fit_size = val_size  # train+val 一起训练
    test_start = fit_size

    fit_mark = data_stamp[:fit_size]
    test_mark = data_stamp[test_start:]

    # 为了严格评估：真实 brent 直接取原序列在“预测点索引”处的值
    # 先用任何一个 IMF 的 test loader 得到预测点全局索引（所有 IMF 完全一致）
    dummy_sig = imfs[0].reshape(-1, 1)
    dummy_test_seg = dummy_sig[test_start:].copy()
    dummy_test_loader = build_loader_univariate_with_index(
        window, pred_len, batch_size,
        dummy_test_seg, test_mark,
        global_start_index=test_start,
        shuffle=False
    )

    all_test_indices = []
    for _x, _y, _xm, _ym, idx in dummy_test_loader:
        all_test_indices.append(idx.numpy())
    all_test_indices = np.concatenate(all_test_indices, axis=0)  # (n_samples,)

    brent_true = series[all_test_indices].reshape(-1, 1)  # 真实 brent（严格对齐）

    # 逐 IMF 训练与预测，然后相加（预测值在 brent 原尺度上相加）
    pred_sum = np.zeros((len(all_test_indices), 1), dtype=float)

    for k in range(K):
        print(f"\n===== Training CASA for IMF {k+1}/{K} =====")
        sig = imfs[k].reshape(-1, 1)

        # scaler：只 fit 在 fit 段（train+val），不碰 test
        scaler = MinMaxScaler()
        sig_scaled = sig.copy()
        sig_scaled[:fit_size] = scaler.fit_transform(sig[:fit_size])
        sig_scaled[fit_size:] = scaler.transform(sig[fit_size:])

        train_seg = sig_scaled[:fit_size]
        test_seg = sig_scaled[test_start:]

        train_loader = build_loader_univariate_with_index(
            window, pred_len, batch_size,
            train_seg, fit_mark,
            global_start_index=0,
            shuffle=True
        )
        test_loader = build_loader_univariate_with_index(
            window, pred_len, batch_size,
            test_seg, test_mark,
            global_start_index=test_start,
            shuffle=False
        )

        config = Config(window, pred_len, batch_size, num_epochs, learning_rate, data_dim=1)
        net = CASA.Model(config).to(device)
        criterion = nn.MSELoss().to(device)
        optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)

        net, _ = model_train(
            net, train_loader, pred_len, optimizer, criterion,
            num_epochs=config.num_epochs, device=device, print_train=True
        )

        # 测试集推理（顺序与 all_test_indices 一致，因为 test_loader shuffle=False）
        y_pred_list = []
        for bx, by, bxm, bym, _idx in test_loader:
            _t, p = model_predict(net, bx, bxm, by, bym, config.label_len, pred_len, device)
            y_pred_list.append(p[:, -1:, 0])  # (B,1)

        y_pred = np.concatenate(y_pred_list, axis=0).reshape(-1, 1)  # (n_samples,1)

        # 反归一化回 IMF 原尺度，然后累加
        y_pred_inv = scaler.inverse_transform(y_pred)
        pred_sum += y_pred_inv

    # =========================
    # 评估：Σ(IMF_pred) vs 真实 brent
    # =========================
    df_eval = cal_eval(brent_true, pred_sum)
    print("\n===== VMD-CASA (Sum of IMF preds -> Brent) Evaluation =====")
    print(df_eval)

    plt.figure(figsize=(12, 4))
    plt.plot(brent_true.flatten(), label="Real brent")
    plt.plot(pred_sum.flatten(), label="Predict brent (sum of IMF preds)")
    plt.title("VMD-CASA Result (Target=brent, IMF-wise prediction & sum)")
    plt.legend()
    plt.show()

    result_df = pd.DataFrame({
        "真实值(brent)": brent_true.flatten(),
        "预测值(brent)": pred_sum.flatten()
    })
    out_path = "brent_真实值与预测值_VMD-CASA_IMFsum.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n已保存：{out_path}")


if __name__ == "__main__":
    main()
