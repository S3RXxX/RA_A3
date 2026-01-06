import math
from scipy.integrate import quad
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm


SEED = 373
seeds = [13323, 60340, 19128, 48828, 14150, 40817, 31592, 58201, 27433, 35026]
seeds2= [4683928, 9059263, 6659057, 8599912, 6323675, 1163603, 3220161,
       5745376, 7787213, 4391116, 7937650, 4048067, 5664925, 6273507,
       9889776, 6068579,  729458, 2075370, 8628342,  880786, 9069474,
       6920074, 4127782, 9661206, 9326793, 6780713, 8134421, 6939713,
       1114846, 1951722, 3861880, 4387187,  781746, 1501138, 5863108,
       7277374, 8214566, 8273103, 3726385, 2661968, 9769767, 9718881,
       3652923, 5886597, 6044068,  857268, 3382833, 6501747, 5312752,
       6013274, 1959420, 4145034, 8963092, 6551589, 6398327, 5262376,
       5220220, 7065820, 6067698,  462585, 9238576, 8125388, 8248717,
       4565770, 6170095, 9011574, 8690230, 2594170, 7062587, 1875105,
       9343188,   97684, 6518793, 2850271, 9365430, 8878321, 8885308,
       2780192, 7970313, 2822398, 9065200,  368790, 8622920, 9058560,
       6315655, 7886155, 6382731, 2932005, 4118409, 8814776, 9085805,
       4308943, 1522524, 4062031,  744616, 6722546, 5447482,  219765,
       1249326,  636270]

HASH_PREDICTORS = ["PCSA", "LogLog", "HLL", "AdaptiveSampling"]
ORDER_PREDICTORS = ["REC", "KMV", "MinCount"]

results_path = "./results/"
datasets_path = "./datasets/"
datasets = ["crusoe", "dracula", "iliad", "mare-balena", "midsummer-nights-dream",
            "quijote", "valley-fear", "war-peace"]
synthetic_datasets = ['synthetic_1000000_100000_1.0', 'synthetic_1000000_10000_0.0', 'synthetic_1000000_10000_0.5', 'synthetic_1000000_10000_1.0', 'synthetic_1000000_10000_1.5', 'synthetic_1000000_10000_50.0', 'synthetic_100000_10000_0.0', 'synthetic_100000_10000_0.5', 'synthetic_100000_10000_1.0', 'synthetic_100000_10000_1.5', 'synthetic_100000_10000_50.0', 'synthetic_100000_1000_0.0', 'synthetic_100000_1000_0.5', 'synthetic_100000_1000_1.0', 'synthetic_100000_1000_1.5', 'synthetic_100000_1000_50.0', 'synthetic_10000_1000_0.0', 'synthetic_10000_1000_0.5', 'synthetic_10000_1000_1.0', 'synthetic_10000_1000_1.5', 'synthetic_10000_1000_50.0', 'synthetic_2500000_50000_0.0', 'synthetic_2500000_50000_0.5', 'synthetic_2500000_50000_1.0', 'synthetic_2500000_50000_1.5', 'synthetic_2500000_50000_50.0', 'synthetic_250000_50000_0.0', 'synthetic_250000_50000_0.5', 'synthetic_250000_50000_1.0', 'synthetic_250000_50000_1.5', 'synthetic_250000_50000_50.0', 'synthetic_250000_5000_0.0', 'synthetic_250000_5000_0.5', 'synthetic_250000_5000_1.0', 'synthetic_250000_5000_1.5', 'synthetic_250000_5000_50.0', 'synthetic_25000_5000_0.0', 'synthetic_25000_5000_0.5', 'synthetic_25000_5000_1.0', 'synthetic_25000_5000_1.5', 'synthetic_25000_5000_50.0', 'synthetic_5000000_50000_0.0', 'synthetic_5000000_50000_0.5', 'synthetic_5000000_50000_1.0', 'synthetic_5000000_50000_1.5', 'synthetic_5000000_50000_50.0', 'synthetic_500000_10000_0.0', 'synthetic_500000_10000_0.5', 'synthetic_500000_10000_1.0', 'synthetic_500000_10000_1.5', 'synthetic_500000_10000_50.0', 'synthetic_500000_50000_0.0', 'synthetic_500000_50000_0.5', 'synthetic_500000_50000_1.0', 'synthetic_500000_50000_1.5', 'synthetic_500000_50000_50.0', 'synthetic_500000_5000_0.0', 'synthetic_500000_5000_0.5', 'synthetic_500000_5000_1.0', 'synthetic_500000_5000_1.5', 'synthetic_500000_5000_50.0', 'synthetic_50000_10000_0.0', 'synthetic_50000_10000_0.5', 'synthetic_50000_10000_1.0', 'synthetic_50000_10000_1.5', 'synthetic_50000_10000_50.0', 'synthetic_50000_1000_0.0', 'synthetic_50000_1000_0.5', 'synthetic_50000_1000_1.0', 'synthetic_50000_1000_1.5', 'synthetic_50000_1000_50.0', 'synthetic_50000_5000_0.0', 'synthetic_50000_5000_0.5', 'synthetic_50000_5000_1.0', 'synthetic_50000_5000_1.5', 'synthetic_50000_5000_50.0', 'synthetic_5000_1000_0.0', 'synthetic_5000_1000_0.5', 'synthetic_5000_1000_1.0', 'synthetic_5000_1000_1.5', 'synthetic_5000_1000_50.0']

def to_bin(x, size=32):
    """Returns binary representation of x with size as the string length"""
    s = bin(x)[2:]
    return (size-len(s))*"0"+s


def leading_zeros_from_bin(b) -> int:
    """
    Count leading zeros before the first '1' in the binary representation.
    """
    count = 0
    for bit in b:
        if bit == '0':
            count += 1
        else:
            break
    return count


def rho(z, size=28):
    """returns the position of the first 1-bit in z: ranks start at 1"""
    assert z.bit_length() <= size
    if z==0:
        return size+1
    
    b = to_bin(z, size=size)
    # print(b)
    return leading_zeros_from_bin(b=b)+1

def J0(m, T=50.0):
    """
    Exact computation of J0(m) via numerical integration.
    """
    def integrand(x):
        return (math.log2((2+x)/(1+x))) ** m

    val, _ = quad(integrand, 0, T, epsabs=1e-12, epsrel=1e-12)
    return val


def execute_save_all(predictors, bs, datasets, output_path, csv_name, seeds, do_hash_REC=True):
    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(output_path, f"{csv_name}.csv")
    all_results = []
    for seed in seeds:
        print(f"seed={seed}")
        for ds in datasets:
            print(f"Data stream: {ds}")
            for b in bs:
                for predictor in predictors:
                    if do_hash_REC:
                        predictor = predictor(b, seed=seed)
                    else:
                        predictor = predictor(b, seed=seed, do_hash=False)
                    predictor.reset()  # just to make sure
                    result = predictor.compute(ds)
                    all_results.append({
                        "Predictor": predictor.__class__.__name__,
                        "b": b,
                        "Seed": seed,
                        "Dataset": ds,
                        "Result": result
                    })

            # save name, b, seed, result in a csv named csv_name.csv and in path output_path
    df = pd.DataFrame(all_results)
    df.to_csv(csv_path, index=False)



def get_val(sub, predictor, b, col):
    v = sub[(sub["Predictor"] == predictor) & (sub["b"] == b)][col]
    return v.iloc[0] if not v.empty else None


def latex_hash_table(df, dataset):
    sub = df[df["Dataset"] == dataset]

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{Hash-based cardinality estimators on the \texttt{{{dataset}}} dataset.}}")
    lines.append(rf"\label{{tab:hash_family_{dataset.replace('-', '_')}}}")
    lines.append(r"\begin{tabular}{rcccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r" & \multicolumn{2}{c}{PCSA}"
        r" & \multicolumn{2}{c}{LogLog}"
        r" & \multicolumn{2}{c}{HyperLogLog}"
        r" & \multicolumn{2}{c}{Adaptive Sampling} \\"
    )
    lines.append(
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}"
        r"\cmidrule(lr){6-7}\cmidrule(lr){8-9}"
    )
    lines.append(r"$b$ & Est. & RSE & Est. & RSE & Est. & RSE & Est. & RSE \\")
    lines.append(r"\midrule")

    for b in sorted(sub["b"].unique()):
        row = [str(b)]
        for p in HASH_PREDICTORS:
            est = get_val(sub, p, b, "mean_estimate")
            rse = get_val(sub, p, b, "RSE")
            if est is None:
                row.extend(["--", "--"])
            else:
                row.extend([f"{est:.0f}", f"{rse:.3f}"])
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def latex_order_table(df, dataset):
    sub = df[df["Dataset"] == dataset]

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{Order-statistics cardinality estimators on the \texttt{{{dataset}}} dataset.}}")
    lines.append(rf"\label{{tab:order_family_{dataset.replace('-', '_')}}}")
    lines.append(r"\begin{tabular}{rcccccc}")
    lines.append(r"\toprule")
    lines.append(
        r" & \multicolumn{2}{c}{REC}"
        r" & \multicolumn{2}{c}{KMV}"
        r" & \multicolumn{2}{c}{MinCount} \\"
    )
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    lines.append(r"$b$ & Est. & RSE & Est. & RSE & Est. & RSE \\")
    lines.append(r"\midrule")

    for b in sorted(sub["b"].unique()):
        row = [str(b)]
        for p in ORDER_PREDICTORS:
            est = get_val(sub, p, b, "mean_estimate")
            rse = get_val(sub, p, b, "RSE")
            if est is None:
                row.extend(["--", "--"])
            else:
                row.extend([f"{est:.0f}", f"{rse:.3f}"])
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def boxplot(df):
    sns.set_style("whitegrid")

    plt.figure(figsize=(10, 6))

    order = [
        "AdaptiveSampling",
        "HLL",
        "KMV",
        "LogLog",
        "MinCount",
        "PCSA",
        "REC",
    ]

    sns.boxplot(
        data=df,
        x="Predictor",
        y="rel_error",
        order=order,
        showfliers=True,
        width=0.6
    )

    plt.axhline(0, color="red", linestyle="--", linewidth=1)

    plt.xlabel("Algorithm")
    plt.ylabel("Relative error $(\\hat{N} - N) / N$")
    plt.title("Signed relative estimation error (all data streams)")

    plt.tight_layout()
    plt.show()



def boxplots(df):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=False)

    axes = axes.flatten()

    order = [
        "AdaptiveSampling",
        "HLL",
        "KMV",
        "LogLog",
        "MinCount",
        "PCSA",
        "REC",
    ]

    for ax, b in zip(axes, range(2, 10)):
        sub = df[df["b"] == b]

        sns.boxplot(
            data=sub,
            x="Predictor",
            y="rel_error",
            order=order,
            ax=ax,
            width=0.6,
            showfliers=True
        )

        # Zero reference line
        ax.axhline(0, color="red", linestyle="--", linewidth=1)

        # --- Set per-panel y-limits ---
        y = sub["rel_error"].dropna()
        lo, hi = np.percentile(y, [1, 99])   # robust against outliers
        pad = 0.1 * (hi - lo)

        ax.set_ylim(lo - pad, hi + pad)

        ax.set_title(f"$b = {b}$")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=60)

    fig.supxlabel("Algorithm")
    fig.supylabel("Relative error $(\\hat{N} - N) / N$")
    fig.suptitle("Signed relative estimation error by memory parameter $b$", y=0.98)

    plt.tight_layout()
    plt.show()



def latex_hll_rec_table(df, dataset):
    sub = df[df["Dataset"] == dataset]

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Cardinality estimates using HyperLogLog and Recordinality on the \texttt{{{dataset}}} dataset.}}"
    )
    lines.append(
        rf"\label{{tab:hll_rec_{dataset.replace('-', '_')}}}"
    )
    lines.append(r"\begin{tabular}{rcccc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{2}{c}{HyperLogLog} & \multicolumn{2}{c}{Recordinality} \\")
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    lines.append(r"$b$ & Estimate & RSE & Estimate & RSE \\")
    lines.append(r"\midrule")

    for b in sorted(sub["b"].unique()):
        hll = sub[(sub["Predictor"] == "HLL") & (sub["b"] == b)]
        rec = sub[(sub["Predictor"] == "REC") & (sub["b"] == b)]

        if not hll.empty:
            hll_est = f"{hll['mean_estimate'].iloc[0]:.0f}"
            hll_rse = f"{hll['RSE'].iloc[0]:.3f}"
        else:
            hll_est, hll_rse = "--", "--"

        if not rec.empty:
            rec_est = f"{rec['mean_estimate'].iloc[0]:.0f}"
            rec_rse = f"{rec['RSE'].iloc[0]:.3f}"
        else:
            rec_est, rec_rse = "--", "--"

        lines.append(
            f"{b} & {hll_est} & {hll_rse} & {rec_est} & {rec_rse} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)




def plot_hll_rec_rse(df, n=9425):
    """
    Plot theoretical vs empirical RSE for HLL and REC.

    Parameters:
    - df: pandas DataFrame with columns Predictor, b, Dataset, RSE
    - n: true cardinality (for REC theoretical calculation)
    """

    # ---- Theoretical functions ----
    theoretical_hll = lambda b: 1.04 / math.sqrt(2 ** b)
    theoretical_rec = lambda b: math.sqrt((n / (math.e * 2 ** b)) ** (1 / (2 ** b)) - 1)

    # ---- Filter HLL and aggregate ----
    hll = df[df["Predictor"] == "HLL"]
    hll_agg = (
        hll.groupby("b", as_index=False)["RSE"].mean().sort_values("b")
    )
    hll_agg["m"] = 2 ** hll_agg["b"]
    hll_agg["RSE_theoretical"] = hll_agg["b"].apply(theoretical_hll)

    # ---- Filter REC and aggregate ----
    rec = df[df["Predictor"] == "REC"]
    rec_agg = (
        rec.groupby("b", as_index=False)["RSE"].mean().sort_values("b")
    )
    rec_agg["m"] = 2 ** rec_agg["b"]
    rec_agg["RSE_theoretical"] = rec_agg["b"].apply(theoretical_rec)

    # ---- Plot ----
    plt.figure(figsize=(8, 6))

    # HLL lines
    plt.plot(
        hll_agg["m"],
        hll_agg["RSE_theoretical"],
        marker="o",
        linestyle="--",
        color="blue",
        label="HLL theoretical"
    )
    plt.plot(
        hll_agg["m"],
        hll_agg["RSE"],
        marker="s",
        linestyle="-",
        color="blue",
        label="HLL empirical"
    )

    # REC lines
    plt.plot(
        rec_agg["m"],
        rec_agg["RSE_theoretical"],
        marker="o",
        linestyle="--",
        color="green",
        label="REC theoretical"
    )
    plt.plot(
        rec_agg["m"],
        rec_agg["RSE"],
        marker="s",
        linestyle="-",
        color="green",
        label="REC empirical"
    )

    # ---- Axes and style ----
    plt.xscale("log", base=2)
    plt.xlabel("Number of registers $m = 2^b$")
    plt.ylabel("Relative standard error (RSE)")
    plt.title("RSE: HLL vs REC (theory vs simulation)")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()



def rel_error_histogram(df, predictor="HLL", b=5, n=9425):
    """
    Plot histogram of relative errors with theoretical normal curve overlay.

    Parameters:
    - df: pandas DataFrame with columns Predictor, b, Dataset, rel_error
    - predictor: "HLL" or "REC"
    - b: memory parameter
    - n: true cardinality (used for REC theoretical std)
    """

    # Filter the dataframe
    sub = df[(df["Predictor"] == predictor) & (df["b"] == b)]

    if predictor == "HLL":
        # Theoretical RSE for HLL
        theoretical_std = 1.04 / np.sqrt(2 ** b)
    elif predictor == "REC":
        # Theoretical RSE for REC
        theoretical_std = np.sqrt((n / (np.e * 2 ** b)) ** (1 / (2 ** b)) - 1)
    else:
        raise ValueError("Predictor must be 'HLL' or 'REC'")

    # ---- Plot histogram ----
    plt.figure(figsize=(7, 5))
    
    # Histogram of relative errors
    sns.histplot(sub["rel_error"], bins=20, kde=False, stat="density", color="skyblue", edgecolor="k", label="Simulation")

    # Overlay normal curve (extend x-range to avoid cropping)
    x_min = min(sub["rel_error"].min(), -4*theoretical_std)
    x_max = max(sub["rel_error"].max(), 4*theoretical_std)
    x = np.linspace(x_min, x_max, 500)
    y = norm.pdf(x, loc=0, scale=theoretical_std)
    plt.plot(x, y, color="red", linewidth=2, label="Theoretical Normal")

    # Labels and title
    plt.xlabel("Relative error $(\\hat{N}-N)/N$")
    plt.ylabel("Density")
    plt.title(f"Histogram of relative errors for {predictor}, b={b}")
    plt.legend()
    plt.grid(True, linestyle=":", linewidth=0.7)
    plt.tight_layout()
    plt.show()




def rel_error_density(df, b=5, n=9425):
    """
    Plot side-by-side density plots of relative errors for HLL and REC
    with theoretical normal curve overlays.

    Parameters:
    - df: pandas DataFrame with columns Predictor, b, Dataset, rel_error
    - b: memory parameter
    - n: true cardinality (used for REC theoretical std)
    """

    predictors = ["HLL", "REC"]

    plt.figure(figsize=(14, 5))  # wider figure for side-by-side plots

    for i, predictor in enumerate(predictors, 1):
        # Filter dataframe
        sub = df[(df["Predictor"] == predictor) & (df["b"] == b)]

        # Theoretical standard deviation
        if predictor == "HLL":
            theoretical_std = 1.04 / np.sqrt(2 ** b)
        else:  # REC
            theoretical_std = np.sqrt((n / (np.e * 2 ** b)) ** (1 / (2 ** b)) - 1)

        # Side-by-side subplot
        plt.subplot(1, 2, i)

        # KDE of simulated relative errors
        sns.kdeplot(
            sub["rel_error"], 
            color="skyblue", 
            linewidth=2, 
            label="Simulation", 
            fill=True, 
            alpha=0.4
        )

        # Theoretical normal curve
        x_min = min(sub["rel_error"].min(), -4*theoretical_std)
        x_max = max(sub["rel_error"].max(), 4*theoretical_std)
        x = np.linspace(x_min, x_max, 500)
        y = norm.pdf(x, loc=0, scale=theoretical_std)
        plt.plot(x, y, color="red", linewidth=2, label="Theoretical Normal")

        # Labels, title, and grid
        plt.xlabel("Relative error $(\\hat{N}-N)/N$")
        plt.ylabel("Density")
        plt.title(f"{predictor}, b={b}")
        plt.legend()
        plt.grid(True, linestyle=":", linewidth=0.7)

    plt.tight_layout()
    plt.show()


def plot_density_subgrid(df, b_values=range(2, 9), n_cols=3):
    """
    Plot KDEs of relative errors for Hash vs No Hash for each memory parameter b
    using a subgrid layout with properly spaced titles.

    Parameters:
    - df: DataFrame with columns Predictor, b, Seed, Dataset, Result, True, Hash
    - b_values: list or range of b values to plot
    - n_cols: number of columns in the grid
    """

    # Compute relative error
    df['rel_error'] = (df['Result'] - df['True']) / df['True']

    n_b = len(b_values)
    n_rows = int(np.ceil(n_b / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharex=False, sharey=False)
    axes = axes.flatten()  # flatten for easy indexing

    for ax, b in zip(axes, b_values):
        sub = df[df['b'] == b]

        sns.kdeplot(
            data=sub,
            x='rel_error',
            hue='Hash',
            fill=True,
            alpha=0.4,
            linewidth=2,
            ax=ax
        )
        # Move title up using pad
        ax.set_title(f"b = {b}", fontsize=12, pad=10)  
        ax.set_xlabel("Relative error $(\\hat{N}-N)/N$", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.grid(True, linestyle=":", linewidth=0.7)

    # Remove any unused axes
    for ax in axes[n_b:]:
        fig.delaxes(ax)

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.3)  

    plt.show()




if __name__=="__main__":
    # print("0", to_bin(0), len(to_bin(0)))
    # print("1", to_bin(1), len(to_bin(1)))
    # print("2", to_bin(2), len(to_bin(2)))
    # print("1+2**9", to_bin(1+2**9), len(to_bin(1+2**9)))
    # print("9999999999999", to_bin(999999999), to_bin(999999999), (999999999).bit_length())
    print(leading_zeros_from_bin("0011"))
    print(leading_zeros_from_bin("0001"))
