# PEDA
Official Repo for [Pareto-Efficient Decision Agents for Offline Multi-Objective Reinforcement Learning](https://openreview.net/forum?id=Ki4ocDm364). Published in ICLR 2023.

Website: [COMING SOON]().

Authors: [Baiting Zhu](https://baitingzbt.github.io/), [Meihua Dang](http://web.cs.ucla.edu/~mhdang/), [Aditya Grover](https://aditya-grover.github.io/)

## Setup
  ```
  conda env create -f environment.yml
  conda activate peda_env
  ```
## Data Download
```
pip install gdown
gdown --folder COMING_SOON --output data
mv data PEDA
cd PEDA
```
The "data" folder should be under "PEDA", with all dataset stored.
## Training
Open the shell script below to double-check your CUDA devices and data path. Then run:
```
sh all_env_uniform.sh
```
## Citation
If you use this repo, please cite:
```
@inproceedings{
zhu2023paretoefficient,
title     = {Pareto-Efficient Decision Agents for Offline Multi-Objective Reinforcement Learning},
author    = {Baiting Zhu and Meihua Dang and Aditya Grover},
booktitle = {International Conference on Learning Representations},
year      = {2023},
url       = {https://openreview.net/forum?id=Ki4ocDm364}
}
```
