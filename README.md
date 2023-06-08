# PEDA
Official Repo for **Scaling Pareto-Efficient Decision Making via Offline Multi-Objective RL**. Published in ICLR 2023.

[Website](https://baitingzbt.github.io/projects/iclr_2023_morl/) | [Poster](https://drive.google.com/file/d/1kiUYbYcfAdd8wLLK7x26NSYCqfWk6mGr/view) | [OpenReview](https://openreview.net/forum?id=Ki4ocDm364)

Authors: [Baiting Zhu](https://baitingzbt.github.io/), [Meihua Dang](http://web.cs.ucla.edu/~mhdang/), [Aditya Grover](https://aditya-grover.github.io/)

## Setup
  ```
  git clone https://github.com/baitingzbt/PEDA.git
  cd PEDA
  conda env create -f environment.yml
  conda activate peda_env
  ```

## Data Download
Due to large storage space, not all dataset variants are stored in this link. However, what's in the folder contain all dataset variants used in the paper experiments including ablation study.
```
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1wfd6BwAu-hNLC9uvsI1WPEOmPpLQVT9k?usp=sharing --output data
```
The "data" folder should be under "PEDA" e.g.: `PEDA/data/env/data_name.pkl`
## Training
First double-check your CUDA devices and data path in this shell script. Run the uniform experiments for all environments:
```
sh all_env_uniform.sh
```
Alternatively, here is an example for a single experiment:
```
python experiment.py --dir experiment_runs/uniform --env MO-HalfCheetah-v2 --data_mode _formal --concat_state_pref 1 --concat_rtg_pref 0 --concat_act_pref 0 --mo_rtg True --seed 1 --dataset expert_uniform --model_type rvs --num_steps_per_iter 200000 --max_iters 2
```
## Generate Your Own Data (WIP)
Due to storage limit, we cannot easily open-source all data variants. Please check the source code to collect data. First download ckpts from `https://drive.google.com/file/d/19kEqdNG-ttwxmZ__30gop_KRvPf4NSjL/view`. Unzip, rename to "Precomputed_Result", and move folder under `data_generation`.
```
# DOWNLOAD, UPZIP, RENAME, MOVE

# USE
cd data_generation
sh collect_all.sh
```
Note 1: We use randomly-initialized environments versus a fixed version in the behavioral policy paper. This helps to diversify trajectories.

Note 2: Model ckpts are stored under `PEDA/data_generation/Precomputed_Results`. All were kindly provided by the authors of behavioral policy paper, except that we trained Hopper-v3 ourselves.
## Citation
If you use this repo, please cite:
```
@inproceedings{
    zhu2023paretoefficient,
    title     = {Scaling Pareto-Efficient Decision Making via Offline Multi-Objective RL},
    author    = {Baiting Zhu and Meihua Dang and Aditya Grover},
    booktitle = {International Conference on Learning Representations},
    year      = {2023},
    url       = {https://openreview.net/forum?id=Ki4ocDm364}
}
```
