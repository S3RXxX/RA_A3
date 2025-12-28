import random
import bisect
import itertools
import os



class DataStream:
    def __init__(self, n, alpha, seed=None):
        self.seed=seed
        self.n = n
        self.alpha = alpha
        # self.rng = random.Random(seed)


    def zipf_distribution(self):
        """
        Returns cumulative distribution function (CDF)
        for Zipf(n, alpha).
        """
        weights = [i ** (-self.alpha) for i in range(1, self.n + 1)]
        Z = sum(weights)
        probs = [w / Z for w in weights]
        # print(probs)
        cdf = list(itertools.accumulate(probs))
        return cdf


    def sample_zipf(self, cdf, rng):
        """
        Draw a single Zipf-distributed index in {1,...,n}.
        """
        u = rng.random()
        return bisect.bisect_left(cdf, u) + 1


    def generate_zipf_stream(self, N):
        """
        Generate a Zipfian stream Z of length N
        with n distinct elements and parameter alpha.
        """
        rng = random.Random(self.seed)
        cdf = self.zipf_distribution()

        stream = []
        for _ in range(N):
            i = self.sample_zipf(cdf, rng)
            stream.append(f"x{i}")

        return stream
    
    def generate_zipf_stream_to_file(self, N, output_path):
        """
        Generate a Zipfian stream of length N and write it to a file.
        """
        rng = random.Random(self.seed)
        cdf = self.zipf_distribution()

        with open(output_path, "w", encoding="utf-8") as f:
            for _ in range(N):
                i = self.sample_zipf(cdf, rng)
                f.write(f"x{i}\n")
    

if __name__=="__main__":
    N = 1_000_000
    n = 100_000
    alpha = 1.0
    seed = 373

    datasets_dir="datasets"
    filename = f"synthetic_{N}_{n}_{alpha}.txt"
    output_path = os.path.join(datasets_dir, filename)

    dataGen = DataStream(n=n, alpha=alpha, seed=seed)
    dataGen.generate_zipf_stream_to_file(N, output_path)

    print(f"Synthetic dataset written to {output_path}")

    # dataGen = DataStream(n=n, alpha=alpha, seed=seed)
    # Z = dataGen.generate_zipf_stream(N)

    # for z in Z:
    #     print(z)



    