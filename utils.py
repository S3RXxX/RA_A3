import math
from scipy.integrate import quad
import os
import pandas as pd

SEED = 373
seeds = [13323, 60340, 19128, 48828, 14150, 40817, 31592, 58201, 27433, 35026]

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





if __name__=="__main__":
    # print("0", to_bin(0), len(to_bin(0)))
    # print("1", to_bin(1), len(to_bin(1)))
    # print("2", to_bin(2), len(to_bin(2)))
    # print("1+2**9", to_bin(1+2**9), len(to_bin(1+2**9)))
    # print("9999999999999", to_bin(999999999), to_bin(999999999), (999999999).bit_length())
    print(leading_zeros_from_bin("0011"))
    print(leading_zeros_from_bin("0001"))
