from HLL import HLL
from utils import results_path, datasets, synthetic_datasets, plot_density_subgrid, latex_hash_table, latex_order_table, boxplot, boxplots, latex_hll_rec_table, plot_hll_rec_rse, rel_error_histogram, rel_error_density
import pandas as pd
from trueCardinality import TrueCardinality
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import boxplot_grid_by_Nmult, boxplot_grid_by_alpha, boxplot_grid_by_n


### True Values
true_values = {}
trueCard = TrueCardinality()
for data in datasets:
    trueCard.reset()
    val = trueCard.compute(data)
    # print(val, data)
    true_values[data] = val

# for data in synthetic_datasets:
#     trueCard.reset()
#     val = trueCard.compute(data)
#     true_values[data] = val

##########################################################################################
####    Experiment 1
##########################################################################################

df = pd.read_csv(results_path+"allComparison.csv")
# header: Predictor,b,Seed,Dataset,Result
df["True"] = df["Dataset"].map(true_values)
df["rel_error"] = (df["Result"] - df["True"]) / df["True"] # for the boxplot, NOT the RSE

boxplot(df=df[df["b"]==5])

boxplot(df=df[df["b"]==8])
boxplot(df=df[df["b"]==9])

boxplots(df=df)


tables = (
    df
    .groupby(["Predictor", "b", "Dataset"])
    .agg(
        mean_estimate=("Result", "mean"),
        variance=("Result", "var"),
        true_value=("True", "first"),
        runs=("Result", "count"),
    )
    .reset_index()
)

# Relative Standard Error
tables["RSE"] = (tables["variance"] ** 0.5) / tables["true_value"]

# ---- Generate tables for all datasets ----
for dataset in sorted(tables["Dataset"].unique()):
    print(latex_hash_table(tables, dataset))
    print()
    print(latex_order_table(tables, dataset))
    print("\n" + "="*80 + "\n")






##########################################################################################
####    Experiment 4
##########################################################################################
df_mem = pd.read_csv(results_path+"draculaMemory2.csv")
df_mem["True"] = df_mem["Dataset"].map(true_values)
df_mem["rel_error"] = (df_mem["Result"] - df_mem["True"]) / df_mem["True"] # for the boxplot, NOT the RSE


########### boxplots(df=df_mem)


tables = (
    df_mem
    .groupby(["Predictor", "b", "Dataset"])
    .agg(
        mean_estimate=("Result", "mean"),
        variance=("Result", "var"),
        true_value=("True", "first"),
        runs=("Result", "count"),
    )
    .reset_index()
)

# Relative Standard Error
tables["RSE"] = (tables["variance"] ** 0.5) / tables["true_value"]

# ---- Generate tables for all datasets ----
for dataset in sorted(tables["Dataset"].unique()):
    print(latex_hll_rec_table(tables, dataset))
    print("\n" + "="*80 + "\n")


# theoretical_by_m = lambda b: 1.04/math.sqrt(2**b)

# n=9425
# theoretical_by_k = lambda b: math.sqrt((n/(math.e*2**b))**(1/(2**b))-1)

plot_hll_rec_rse(df=tables[tables["b"]<10])

## rel_error_histogram(df=df_mem, predictor="REC", b=2)
rel_error_density(df=df_mem, b=2)
## rel_error_histogram(df=df_mem, predictor="REC", b=4)
rel_error_density(df=df_mem, b=4)
## rel_error_histogram(df=df_mem, predictor="REC", b=8)
rel_error_density(df=df_mem, b=8)


##########################################################################################
####    Experiment 3
##########################################################################################

# line plot RSE hash vs no hash (amb variancia)
# algun histograma
df_hash = df[(df["Predictor"]=="REC")&(df["b"]<9)].copy()
df_no = pd.read_csv(results_path+"RecNoHash.csv")
df_no["True"] = df_no["Dataset"].map(true_values)
df_no["rel_error"] = (df_no["Result"] - df_no["True"]) / df_no["True"] # for the boxplot, NOT the RSE

df_hash['Hash'] = 'With Hash'
df_no['Hash'] = 'No Hash'

df_h = pd.concat([df_hash, df_no], ignore_index=True)




plot_density_subgrid(df_h)

agg_df = df_h.groupby(['b', 'Hash']).agg(
    mean_rel_error=('rel_error', 'mean'),
    std_rel_error=('rel_error', 'std')
).reset_index()




plt.figure(figsize=(8, 5))

for hash_type in agg_df['Hash'].unique():
    subset = agg_df[agg_df['Hash'] == hash_type]
    plt.plot(subset['b'], subset['mean_rel_error'], marker='o', label=hash_type)
    plt.fill_between(
        subset['b'],
        subset['mean_rel_error'] - subset['std_rel_error'],
        subset['mean_rel_error'] + subset['std_rel_error'],
        alpha=0.2
    )

plt.xlabel("Memory parameter b")
plt.ylabel("Mean relative error $(\\hat{N}-N)/N$")
plt.title("Relative error vs Memory parameter (Hash vs No Hash)")
plt.grid(True, linestyle=":", linewidth=0.7)
plt.legend()
plt.tight_layout()
plt.show()

##########################################################################################
####    Experiment 2
##########################################################################################
true_values_syn = {}
for data in synthetic_datasets:
    trueCard.reset()
    val = trueCard.compute(data)
    true_values_syn[data] = val



rows = []
for key, value in true_values_syn.items():
    parts = key.split('_')
    N = int(parts[1])
    n = int(parts[2])
    alpha = float(parts[3])
    N_mult = N / n  # N multiplier
    rows.append({'N': N, 'n': n, 'alpha': alpha, 'true_cardinality': value, 'N_mult': N_mult})

df = pd.DataFrame(rows)

# ---- Plot ----
unique_alphas = sorted(df['alpha'].unique())
n_cols = 3
n_rows = int(np.ceil(len(unique_alphas)/n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharex=False, sharey=False)
axes = axes.flatten()

for ax, alpha in zip(axes, unique_alphas):
    sub = df[df['alpha'] == alpha]
    sns.lineplot(
        data=sub,
        x='n',
        y='true_cardinality',
        hue='N_mult',
        marker='o',
        linewidth=2,
        ax=ax
    )
    ax.set_title(f"alpha = {alpha}", fontsize=12, pad=15)
    ax.set_xlabel("n (number of unique elements)", fontsize=10)
    ax.set_ylabel("True cardinality", fontsize=10)
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend(title="N/n multiplier", loc='upper left', fontsize=9)

# Remove unused axes
for ax in axes[len(unique_alphas):]:
    fig.delaxes(ax)

plt.tight_layout()
plt.show()


df = pd.read_csv(results_path+"syntheticData.csv")
### Predictor,b,Seed,Dataset,Result

### add N,n,alpha, N_mult
def parse_dataset(name):
    _, N, n, alpha = name.split("_")
    return int(N), int(n), float(alpha)

df[['N', 'n', 'alpha']] = df['Dataset'].apply(
    lambda x: pd.Series(parse_dataset(x))
)

df['N_mult'] = df['N'] / df['n']
df['True'] = df['Dataset'].map(true_values_syn)
df['rel_error'] = (df['Result'] - df['True']) / df['True']

boxplot_grid_by_n(df)

boxplot_grid_by_alpha(df)
boxplot_grid_by_Nmult(df)