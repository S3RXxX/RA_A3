import math
from scipy.integrate import quad

SEED = 373
SEEDS = []

datasets_path = "./datasets/"
datasets = ["crusoe", "dracula", "iliad", "mare-balena", "midsummer-nights-dream",
            "quijote", "valley-fear", "war-peace"]
datasets = ["synthetic_1000000_100000_1.0"]

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




if __name__=="__main__":
    # print("0", to_bin(0), len(to_bin(0)))
    # print("1", to_bin(1), len(to_bin(1)))
    # print("2", to_bin(2), len(to_bin(2)))
    # print("1+2**9", to_bin(1+2**9), len(to_bin(1+2**9)))
    # print("9999999999999", to_bin(999999999), to_bin(999999999), (999999999).bit_length())
    print(leading_zeros_from_bin("0011"))
    print(leading_zeros_from_bin("0001"))
