#!/usr/local_rwth/bin/zsh
### ask for 10 GB memory
#SBATCH --mem-per-cpu=8G   #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)
### name the job
#SBATCH --job-name=basic_C0.1_P_0.2_A1
### job run time
#SBATCH --time=5:00:00
### declare the merged STDOUT/STDERR file
#SBATCH --output=basic_FL_logs.%J.txt
###
#SBATCH --mail-type=ALL
###
#SBATCH --mail-user=praveen.yadav@rwth-aachen.de
### request a GPU
#SBATCH --gres=gpu:pascal:1

### begin of executable commands
cd $HOME/repos/Playground/federated_learning
### load modules
module switch intel gcc
module load python/3.8.7
module load cuda/111
module load cudnn/8.0.5
# pip3 install --user tensorboard
# pip3 install --user -U scikit-learn

python3 FL_basic.py -m_sel basic -c_frac 0.1 -p_frac 0.2 -att 1 -t_cls 100 -lr 0.001 -bs 32 -fdrs 400 -leps 3 --jlog -tevr 10
