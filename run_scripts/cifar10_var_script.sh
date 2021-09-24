#!/usr/local_rwth/bin/zsh
### ask for 8 GB memory
#SBATCH --mem-per-cpu=8G   #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)
### name the job
#SBATCH --job-name=cifar10
### job run time
#SBATCH --time=40:00:00
### declare the merged STDOUT/STDERR file
#SBATCH --output=cifar10_logs.%J.txt
### request a GPU
#SBATCH --gres=gpu:pascal:1

### begin of executable commands
cd $HOME/repos/CosDefence/federated_learning
### load modules
module switch intel gcc
module load python
module load cuda/111
module load cudnn/8.0.5
# pip3 install --user tensorboard
# pip3 install --user -U scikit-learn
# pip3 install plotly
# pip3 install -U kaleido
# pip3 install pyyaml

python3 run_config_variations.py cifar10_modified.yaml
