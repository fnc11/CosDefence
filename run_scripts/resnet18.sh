#!/usr/local_rwth/bin/zsh
### ask for 20 GB memory
#SBATCH --mem-per-cpu=20G   #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)
### name the job
#SBATCH --job-name=gaps
### job run time
#SBATCH --time=01:00:00
### declare the merged STDOUT/STDERR file
#SBATCH --output=resnet18_FL_logs.%J.txt
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

python3 FL_basic.py -m_sel resnet18 -c_frac 0.2 -t_cls 100 -lr 0.001 -bs 32 -fdrs 400 -leps 3  --log -tevr 10 -ccds
