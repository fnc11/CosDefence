#!/usr/local_rwth/bin/zsh

for i in {1..32}
do
   sbatch basic_model($i).sh
done