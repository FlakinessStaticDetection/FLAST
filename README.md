# Know Your Neighbor: Fast Static Prediction of Test Flakiness

This repository is a companion page for the submission "Know Your Neighbor: Fast Static Prediction of Test Flakiness".

It contains all the material required for replicating the experiments, including: the algorithm implementation, the datasets and their ground truth, and the scripts for the experiments replication.

It also contains additional material used to investigate the effect of FLAST's parameter on its effectiveness and efficiency.


Experiment Replication
---------------
In order to replicate the experiment follow these steps:

### Getting started

1. Clone the repository:
   - `git clone https://github.com/FlakinessStaticDetection/FLAST`
 
2. The experiments have been run using *python3.9.2*. You can get the appropriate version for your OS [here](https://www.python.org/downloads/).

3. Install the additional python packages required:
   - `python3 -m pip install -r requirements.txt`

### Dataset creation
Decompress the dataset:
   - `tar zxvf dataset.tgz`
   
### Answering the Research Questions
Execute the research questions scripts.

##### RQ1 and RQ2:
   - `python3 py/eff-eff.py`

##### RQ3:
   - `python3 py/compare-pinto.py`


Pseudocode
---------------
The pseudocode of FLAST is available [here](pseudocode/README.md).


Directory Structure
---------------
This is the root directory of the repository. The directory is structured as follows:

    FLAST
     |
     |--- dataset/            Dataset folder, automatically generated after the decompression of `dataset.tgz`.
     |
     |--- manual-inspection/  Tests considered in the manual inspection.
     |
     |--- parameters/         The investigation on the effect of FLAST's parameters.
     |
     |--- pseudocode/         The pseudocode of FLAST.
     |
     |--- py/                 Scripts with FLAST implementation and scripts to run experiments.
     |
     |--- results/            Folder with the results of the experiments.
     
