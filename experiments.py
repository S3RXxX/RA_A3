from HLL import HLL
from REC import REC
from dataStream import DataStream

from utils import results_path, datasets_path, datasets, seeds, execute_save_all



if __name__=="__main__":
    predictors = []

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
    for data in datasets:
        execute_save_all(predictors=predictors, ds=data, output_path=results_path, csv_name="allComparison")


    #####################################################################

    # experiment for alpha value


    ######################################################################


    # Recordinality without hash functions (using items in the data stream)
    ## real data
    # for 

    ## synthetic data


    #####################################################################

    # analysis of the memory (m in HLL, k in REC) datasets:=[dracula.txt] " Check that the standard error behaves as predicted by the theory."






