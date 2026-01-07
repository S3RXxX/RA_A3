Cardinality Estimation on Data Streams

This project implements and evaluates several probabilistic cardinality estimation algorithms for data streams. The goal is to compare their accuracy and behavior under different conditions using synthetic data streams generated according to a Zipfian distribution.

Project Structure.

├── datasets/               # Generated data streams 

├── images/                 # Plots generated during analysis 

├── results/                # CSV files containing experimental results 

│ 

├── adaptiveSampling.py     # Adaptive Sampling estimator 

├── analysis.py             # Generates plots from CSV files in results/ 

├── assignment-cardest.pdf  # Assignment specification 

├── cardinality_estimator.py# Base class / common interface for estimators 

├── dataStream.py           # Zipfian data stream generator 

├── experiments.py          # Runs experiments and saves results as CSV files 

├── HLL.py                  # HyperLogLog estimator 

├── KMV.py                  # K-Minimum Values (KMV) estimator 

├── LogLog.py               # LogLog estimator 

├── MinCount.py             # MinCount estimator 

├── PCSA.py                 # Probabilistic Counting with Stochastic Averaging 

├── REC.py                  # Recordinality estimator 

├── trueCardinality.py      # Exact cardinality computation (ground truth) 

├── utils.py                # Shared utility functions 

├── requirements.txt        # Requirements 

└── .gitignore              # Git ignore rules 


# Overview

Implemented Estimators

Adaptive Sampling

HyperLogLog (HLL)

K-Minimum Values (KMV)

LogLog

MinCount

PCSA

Recordinality (REC)

Each estimator follows a common interface defined in cardinality_estimator.py.

# How to Run

Create a virtual env
python -m venv env

Linux/Macos -> source env/bin/activate

Windows -> .\env\Scripts\Activate

pip install -r requirements.txt


Generate data streams

python dataStream.py


Run experiments

python experiments.py


Generate plots

python analysis.py

Output

Datasets: datasets/

Experimental results (CSV): results/

Plots and figures: images/

