#!/usr/local_rwth/bin/zsh
### ask for 10 GB memory
#SBATCH --mem-per-cpu=10G   #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)
### name the job
#SBATCH --job-name=cifar10_C0.2_P_0.4
### job run time
#SBATCH --time=10:00:00
### declare the merged STDOUT/STDERR file
#SBATCH --output=cifar10_logs.%J.txt
###
#SBATCH --mail-type=ALL
###
#SBATCH --mail-user=praveen.yadav@rwth-aachen.de
### request a GPU
#SBATCH --gres=gpu:pascal:1

### begin of executable commands
cd $HOME/repos/CosDefence/federated_learning
### load modules
module switch intel gcc
module load python/3.8.7
module load cuda/111
module load cudnn/8.0.5
# pip3 install --user tensorboard
# pip3 install --user -U scikit-learn
# pip3 install plotly
# pip3 install -U kaleido

python3 FL_basic.py -c_def -ccds -d_sel cifar10 -c_sep 12.0 -c_frac 0.1 -p_frac 0.4 -lr 0.001 -bs 32 -fdrs 200 -leps 3 --jlog -tevr 3
