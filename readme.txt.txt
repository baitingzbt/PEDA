1. Download the datasets from https://drive.google.com/drive/folders/1FiF5xmCSJ2vL_frLYmeZNc_nUrhRfUXC?usp=sharing
2. Move "processed" under PEDA
3. Run experiment with specified environment and toy data


Example Code:

conda env create -f environment.yml
conda activate peda_env

pip install gdown
gdown --folder https://drive.google.com/drive/folders/1FiF5xmCSJ2vL_frLYmeZNc_nUrhRfUXC?usp=sharing --output processed
mv processed PEDA
cd PEDA

python experiment.py --seed 10000