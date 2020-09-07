# Know Your Neighbor: Fast Static Prediction of Test Flakiness

This repository is a companion page for the submission "Know Your Neighbor: Fast Static Prediction of Test Flakiness".

It contains all the material required for replicating the experiments, including: the algorithm implementation, the datasets and their ground truth, and the scripts for the experiments replication.


Experiment Replication
---------------
In order to replicate the experiment follow these steps:

### Getting started

1. Clone the repository:
   - `git clone https://github.com/FlakyFAST/FLAST`
 
2. If you do not have *python3* installed you can get the appropriate version for your OS [here](https://www.python.org/downloads/).

3. Install the additional python packages required:
   - `python -m pip install -r requirements.txt`

### Dataset creation
Decompress the dataset:
   - `tar zxvf dataset.tgz`
   
### Answering the Research Questions
Execute the research questions scripts.

##### RQ1:
   - `python params-k.py` (varying k)
   - `python params-dist.py` (varying distance)
   - `python params-eps.py` (varying epsilon)
   - `python params-sigma.py` (varying sigma)

##### RQ2:
   - `python training-size.py`

##### RQ3 & RQ4:
   - `python single-projects.py` (RQ3 & RQ4, effectiveness and running time)
   - `python storage.py` (RQ4, storage overhead)
   - `python random-classifier.py` (comparison with random classifier)



Pseudocode
---------------
The pseudocode of FLAST is available [here](pseudocode/README.md).


Directory Structure
---------------
This is the root directory of the repository. The directory is structured as follows:

    FLAST
     .                        Scripts with FLAST implementation and scripts to run experiments.
     |
     |--- dataset/            Dataset folder, automatically generated after the decompression of `dataset.tgz`.
     |
     |--- manual-inspection/  Tests considered in the manual inspection
     |
     |--- pseudocode/         The pseudocode of FLAST.