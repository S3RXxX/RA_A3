from HLL import HLL
from utils import results_path, datasets_path, datasets, seeds, execute_save_all
import pandas as pd

# execute_save_all(predictors=[HLL], bs=[i for i in range(2, 17)], datasets=["dracula"], output_path=results_path, csv_name="test", seeds=seeds)


df = pd.read_csv(results_path+"allComparison.csv")
print(df.describe())

summary = (
    df
    .groupby(["Predictor", "b", "Dataset"])["Result"]
    .agg(["mean", "max", "min"])
    .reset_index()
)

print("summary:\n", summary)

