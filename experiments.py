from HLL import HLL
from REC import REC
from dataStream import DataStream

from utils import datasets_path, datasets, seeds



if __name__=="__main__":

    # check cardinalities using .dat files
    for data in datasets:
        c=0
        with open(datasets_path+data+".dat") as f:
            for word in f.read().splitlines():
                c+=1
        print(f"{data} {c}")

    # HLL REC table comparison estimation vs true cardinalities
    ### Adding more cardinality estimation algorithms 
    # (Probabilistic Counting (PCSA), KMV (K Minimum Values), MinCount, Adaptive Sampling, LogLog) 
    ## real data

    ## synthetic data


    #####################################################################


    # Recordinality without hash functions (using items in the data stream)
    ## real data

    ## synthetic data


    #####################################################################

    # analysis of the memory (m in HLL, k in REC) datasets:=[dracula.txt] " Check that the standard error behaves as predicted by the theory."


    #####################################################################

    # experiment for alpha value


    ######################################################################


