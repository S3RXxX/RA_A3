SEED = 373
SEEDS = []

datasets_path = "./datasets/"
datasets = ["crusoe", "dracula", "iliad", "mare-balena", "midsummer-nights-dream",
            "quijote", "valley-fear", "war-peace"]

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
    """returns the position of the first 1-bit in z: ranks start at 0"""
    assert z.bit_length() <= size
    if z==0:
        return size
    
    b = to_bin(z, size=size)
    return leading_zeros_from_bin(b=b)


if __name__=="__main__":
    print("0", to_bin(0), len(to_bin(0)))
    print("1", to_bin(1), len(to_bin(1)))
    print("2", to_bin(2), len(to_bin(2)))
    print("1+2**9", to_bin(1+2**9), len(to_bin(1+2**9)))
    print("9999999999999", to_bin(999999999), to_bin(999999999), (999999999).bit_length())
