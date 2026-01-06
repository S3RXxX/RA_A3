from HLL import HLL
from REC import REC
from PCSA import PCSA
from KMV import KMV
from MinCount import MinCount
from adaptiveSampling import AdaptiveSampling
from LogLog import LogLog
from utils import results_path, datasets_path, datasets, seeds, execute_save_all, synthetic_datasets, seeds2



if __name__=="__main__":
    predictors = [AdaptiveSampling, PCSA, LogLog, HLL, KMV, MinCount, REC]

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
    execute_save_all(predictors=predictors, bs=[b for b in range(2, 10)], datasets=datasets, output_path=results_path, csv_name="allComparison", seeds=seeds)



    #####################################################################

    # experiment for alpha value
    execute_save_all(predictors=predictors, bs=[b for b in range(5, 10)], datasets=synthetic_datasets, output_path=results_path, csv_name="syntheticData", seeds=seeds)

    ######################################################################


    # Recordinality without hash functions (using items in the data stream)
    predictors3 = [REC]
    ## real data
    execute_save_all(predictors=predictors3, bs=[i for i in range(2,9)], datasets=datasets, output_path=results_path, csv_name="RecNoHash", seeds=seeds, do_hash_REC=False)



    #####################################################################

    # analysis of the memory (m in HLL, k in REC) datasets:=[dracula.txt] " Check that the standard error behaves as predicted by the theory."
    predictors4 = [HLL, REC]
    execute_save_all(predictors=predictors4, bs=[i for i in range(2, 17)], datasets=["dracula"], output_path=results_path, csv_name="draculaMemory", seeds=seeds)
    execute_save_all(predictors=predictors4, bs=[i for i in range(2, 17)], datasets=["dracula"], output_path=results_path, csv_name="draculaMemory2", seeds=seeds2)


