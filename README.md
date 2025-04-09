# Code for "Measuring Social Resilience against Hybrid Threats"
## Introduction
Hybrid threats exploit vulnerabilities across domains, aiming to erode public trust in institutions and authorities and disrupt social stability. 
These threats have become more frequent over the past decade, raising concerns about their impact on society and the ability of individuals to cope with them. 
However, research on applying resilience principles to understand how people can counter these threats remains limited.
## Project structure
```
measuring-social-resilience-against-ht/
├── others-parameters                # Folder contraining trained neural networks for attackers and defenders
├── data                             # Folder containing data on attack strategies
├── parameters                       # Folder containing data and R scripts for hyperparameter tuning
    ├── hyperparameters.csv          # (Tuned) hyperparameter values
    └── parameters.csv               # Parameter values
├── regagent-parameters              # Folder contraining trained neural networks for regular agents
├── results                          # Folder containing R scripts for results analysis and visualisation
    ├── exp1-results                 # Folder containing example data from experiment 1
    ├── exp2-results                 # Folder containing example data from experiment 2
    ├── exp3-results                 # Folder containing example data from experiment 3
    ├── exp1-analysis.R              # R script for analysing and visualising experiment 1 data
    ├── exp2-3-analysis.R            # R script for analysing experiment 2 and 3 data
    └── exp2-3-data-prep.R           # R script for preparing experiment 2 and 3 data for analysis
├── a2c_agent.py                     # Regular agents' behaviour in the environment
├── a2c_def_agent.py                 # Defenders' behaviour in the environment
├── a2c_mal_agent.py                 # Attackers' behaviour in the environment
├── analyse_resilience_exp1.py       # Experiment for measuring social resilience (experiment 1)
├── analyse_resilience_exp2.py       # Experiment for measuring social resilience, attackers in the environment (experiment 2)
├── analyse_resilience_exp3.py       # Experiment for measuring social resilience, attackers and defenders in the environment (experiment 3)
├── data_analysis.py                 # Functions to analyse training data
├── environment_exp1.py              # Environment setup for experiment 1
├── environment_exp2_3.py            # Environment setup for experiment 2 and 3
├── nns.py                           # Architecture of deep neural networks
├── LICENSE.md                       # License
└── README.md                        # Project documentation
```
## License
MIT
## Prerequisites
```
Python 3.10 or higher version is required.

The following Python libraries are required:
- numpy (version 1.24.2 or higher)
- pandas (version 1.5.3 or higher)
- torch (version 1.13.1 or higher)
- matplotlib (version 3.7.0 or higher)
- gymnasium (version 0.29.1 or higher)
- networkx (version 3.0 or higher)
- pettingzoo (version 1.24.1 or higher)
- IPython (version 8.18.1 or higher)

R version 4.4.2 or higher version is required for data analysis and visualisation.

The following R libraries are required:
- psych (version 2.4.12 or higher)
- dplyr (version 1.1.4 or higher)
- tidyverse (version 2.0.0 or higher)
- mcp (version 0.3.4 or higher)
- pracma (version 2.4.4 or higher)
- rjags (version 4.17 or higher)
```
## Run experiments
```
# Run experiment 1
python analyse_resilience_exp1.py
# Run experiment 2
python analyse_resilience_exp2.py
# Run experiment 3
python analyse_resilience_exp3.py
```
## Results
Results will be saved to ```/results``` directory. Experiment 1 results are located in ```/results/exp1-results```, experiment 2 results are located in ```/results/exp2-results```, 
and experiment 3 results are located in ```/results/exp3-results```.

Directory ```/regagent-parameters``` and ```/attacker-defender--parameters``` contains trained deep neural networks.

## Contact
For any questions or issues, please feel free to contact [kart.padur.20@ucl.ac.uk] and I will be happy to assist.
