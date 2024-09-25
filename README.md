# AAPM

This is the official repository for AAPM.



## Installation

1. First clone the directory. 

```code
git submodule init; git submodule update
```
(If showing error of no permission, need to first [add a new SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).)

2. Install dependencies.

Create a new environment using [conda](https://docs.conda.io/en/latest/miniconda.html), with Python >= 3.10.6 Install [PyTorch](https://pytorch.org/) (version >= 2.0.0). The repo is tested with PyTorch version of 1.10.1 and there is no guarentee that other version works. Then install other dependencies via:
```code
pip install -r requirements.txt
```

3. Download news dataset.

Download the WSJ dataset and unzip it.




## How to use?


1. Build the Factor dataset from https://github.com/bkelly-lab/ReplicationCrisis with your CRSP credential, and download the daily return data from https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html for free.

2. Setup your configurations in config.yaml and input your API keys accordingly.
   
3. Use analysis.py to produce the analysis report features.

4. Use model.py to train the hybrid asset pricing model.

