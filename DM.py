import os
import glob
import numpy as np
import pandas as pd
from math import sqrt
from scipy.stats import norm


# ---------------------------
# Robust CSV reader
# ---------------------------
def read_csv_safely(fp: str) -> pd.DataFrame:
    """
    Auto-read CSV with:
    - common encodings (utf-8-sig / gb18030 / gbk / latin1)
    - auto delimiter detection (tab/comma/semicolon/space)
    """
    for enc in ["utf-8-sig", "gb18030", "gbk", "latin1"]:
        try:
            return pd.read_csv(fp, encoding=enc, sep=None, engine="python")
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    return pd.read_csv(fp, encoding="utf-8", sep=None, engine="python")


# ---------------------------
# Newey-West variance
# ---------------------------
def newey_west_var(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    T = x.shape[0]
    if T < 2:
        return np.nan

    x = x - np.mean(x)
    gamma0 = np.dot(x, x) / T

    var_long = gamma0
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)  # Bartlett weight
        cov = np.dot(x[l:], x[:-l]) / T
        var_long += 2.0 * w * cov

    return var_long / T  # variance of sample mean


# ---------------------------
# Diebold-Mariano test
# ---------------------------
def dm_test(y_true, y_pred_ref, y_pred_comp, h=1, loss="MSE", lag=None):
    """
    DM test on out-of-sample errors:
    d_t = L(ref) - L(comp)
    H0: E[d_t] = 0
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_ref = np.asarray(y_pred_ref, dtype=float)
    y_pred_comp = np.asarray(y_pred_comp, dtype=float)

    n = min(len(y_true), len(y_pred_ref), len(y_pred_comp))
    y_true, y_pred_ref, y_pred_comp = y_true[:n], y_pred_ref[:n], y_pred_comp[:n]

    e_ref = y_true - y_pred_ref
    e_comp = y_true - y_pred_comp

    loss = loss.upper().strip()
    if loss == "MSE":
        d = (e_ref ** 2) - (e_comp ** 2)
    elif loss == "MAE":
        d = np.abs(e_ref) - np.abs(e_comp)
    else:
        raise ValueError("loss must be 'MSE' or 'MAE'")

    T = len(d)
    if T < 10:
        return {"T": T, "DM": np.nan, "p": np.nan, "mean_d": float(np.mean(d))}

    if lag is None:
        lag = max(h - 1, 0)

    var_mean = newey_west_var(d, lag=lag)
    if np.isnan(var_mean) or var_mean <= 0:
        return {"T": T, "DM": np.nan, "p": np.nan, "mean_d": float(np.mean(d))}

    dm_stat = np.mean(d) / sqrt(var_mean)
    p_value = 2.0 * (1.0 - norm.cdf(abs(dm_stat)))
    return {"T": T, "DM": float(dm_stat), "p": float(p_value), "mean_d": float(np.mean(d))}


def sig_stars(p):
    if p is None or np.isnan(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def format_p(p: float) -> str:
    """Avoid showing p=0; use scientific threshold."""
    if p is None or np.isnan(p):
        return ""
    if p < 1e-16:
        return "<1e-16"
    return f"{p:.6g}"


# ---------------------------
# Multiple testing corrections
# ---------------------------
def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """
    Holm-Bonferroni adjusted p-values.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)

    running_max = 0.0
    for k, idx in enumerate(order, start=1):
        mult = (m - k + 1) * pvals[idx]
        running_max = max(running_max, mult)
        adj[idx] = min(running_max, 1.0)
    return adj


def bh_fdr_adjust(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR adjusted p-values.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)

    prev = 1.0
    for rank_rev, idx in enumerate(order[::-1], start=1):
        rank = m - rank_rev + 1
        val = (m / rank) * pvals[idx]
        prev = min(prev, val)
        adj[idx] = min(prev, 1.0)
    return adj


# ---------------------------
# Main
# ---------------------------
def main():
    folder = os.path.dirname(os.path.abspath(__file__))

    # ====== read ONLY model prediction csvs in the same folder ======
    all_csv = sorted(glob.glob(os.path.join(folder, "*.csv")))

    # 排除脚本生成的结果文件，避免被当成输入数据
    exclude_prefixes = ("dm_test_results", "error_summary")
    exclude_exact = set()  # 你如果还有固定要排除的文件名，可加进来（小写）

    csv_files = []
    for fp in all_csv:
        base = os.path.basename(fp).lower()
        if base in exclude_exact:
            continue
        if base.startswith(exclude_prefixes):
            continue
        csv_files.append(fp)

    if not csv_files:
        raise FileNotFoundError(
            f"No MODEL prediction CSV files found in folder: {folder}\n"
            f"Found only: {[os.path.basename(x) for x in all_csv]}"
        )

    # ===== settings =====
    reference_keyword = "ALA-VMD-CASA"  # change if needed
    horizon_h = 1                       # 1-step here
    losses = ["MSE", "MAE"]             # repeat DM under both losses

    # ===== load all model files =====
    models = {}
    for fp in csv_files:
        name = os.path.splitext(os.path.basename(fp))[0]
        df = read_csv_safely(fp)
        df.columns = [c.strip() for c in df.columns]

        if "y_true" not in df.columns or "y_pred" not in df.columns:
            raise ValueError(
                f"{fp} must contain columns y_true and y_pred. Found: {list(df.columns)}"
            )

        df = df[["y_true", "y_pred"]].dropna()
        models[name] = df

    # ===== locate reference =====
    ref_candidates = [k for k in models.keys() if reference_keyword.lower() in k.lower()]
    if not ref_candidates:
        raise ValueError(
            f"Reference model not found (filename contains: {reference_keyword}). "
            f"Available: {list(models.keys())}"
        )

    ref_name = ref_candidates[0]
    ref_df = models[ref_name]
    y_true_ref = ref_df["y_true"].to_numpy(dtype=float)
    y_pred_ref = ref_df["y_pred"].to_numpy(dtype=float)

    # ===== output 1: error summary =====
    summary_rows = []
    for name, df in models.items():
        y = df["y_true"].to_numpy(dtype=float)
        yhat = df["y_pred"].to_numpy(dtype=float)
        n = min(len(y), len(yhat))
        y, yhat = y[:n], yhat[:n]

        err = y - yhat
        mse = float(np.mean(err ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(err)))
        mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y), 1e-12)) * 100)

        summary_rows.append({"model": name, "N": n, "MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE(%)": mape})

    summary_df = pd.DataFrame(summary_rows).sort_values(by="MSE")
    summary_path = os.path.join(folder, "error_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # ===== DM tests for both losses =====
    dm_rows = []
    for loss in losses:
        for comp_name, df in models.items():
            if comp_name == ref_name:
                continue
            y_pred_comp = df["y_pred"].to_numpy(dtype=float)

            n = min(len(y_true_ref), len(y_pred_ref), len(y_pred_comp))
            res = dm_test(
                y_true=y_true_ref[:n],
                y_pred_ref=y_pred_ref[:n],
                y_pred_comp=y_pred_comp[:n],
                h=horizon_h,
                loss=loss
            )
            p = res["p"]
            dm_rows.append({
                "reference": ref_name,
                "competitor": comp_name,
                "loss": loss,
                "horizon(h)": horizon_h,
                "N": res["T"],
                "mean_loss_diff(ref - comp)": res["mean_d"],
                "DM_stat": res["DM"],
                "p_value": p,
                "p_value_disp": format_p(p),
                "sig_raw": sig_stars(p),
                "conclusion": "ref better" if res["mean_d"] < 0 else "comp better"
            })

    dm_raw = pd.DataFrame(dm_rows).sort_values(by=["loss", "p_value"])
    raw_path = os.path.join(folder, "dm_test_results_raw.csv")
    dm_raw.to_csv(raw_path, index=False, encoding="utf-8-sig")

    # ===== Multiple testing correction within each loss block =====
    dm_adj = dm_raw.copy()
    dm_adj["p_holm"] = np.nan
    dm_adj["p_fdr_bh"] = np.nan

    for loss in losses:
        mask = dm_adj["loss"].str.upper() == loss
        pvals = dm_adj.loc[mask, "p_value"].to_numpy(dtype=float)

        valid = ~np.isnan(pvals)
        p_holm = np.full_like(pvals, np.nan, dtype=float)
        p_fdr = np.full_like(pvals, np.nan, dtype=float)

        if valid.sum() > 0:
            p_holm[valid] = holm_adjust(pvals[valid])
            p_fdr[valid] = bh_fdr_adjust(pvals[valid])

        dm_adj.loc[mask, "p_holm"] = p_holm
        dm_adj.loc[mask, "p_fdr_bh"] = p_fdr

    dm_adj["p_holm_disp"] = dm_adj["p_holm"].apply(format_p)
    dm_adj["p_fdr_bh_disp"] = dm_adj["p_fdr_bh"].apply(format_p)
    dm_adj["sig_holm"] = dm_adj["p_holm"].apply(sig_stars)
    dm_adj["sig_fdr_bh"] = dm_adj["p_fdr_bh"].apply(sig_stars)

    adj_path = os.path.join(folder, "dm_test_results_adjusted.csv")
    dm_adj.sort_values(by=["loss", "p_value"]).to_csv(adj_path, index=False, encoding="utf-8-sig")

    # ===== Console summary =====
    print("========== DM test finished ==========")
    print(f"Reference model: {ref_name}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {raw_path}")
    print(f"Wrote: {adj_path}")
    print("\nTop rows (per loss) by smallest raw p-values:")
    for loss in losses:
        print(f"\n--- {loss} ---")
        print(dm_adj[dm_adj["loss"] == loss].sort_values("p_value").head(10).to_string(index=False))


if __name__ == "__main__":
    main()
