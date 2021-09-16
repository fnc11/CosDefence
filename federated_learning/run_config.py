import sys
import os
import yaml

from FL_basic import start_fl


global base_path
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def print_results(config, experiment_results):
    mean_poison_class_acc, mean_avg_acc, mean_poison_class_f1_score, mean_avg_f1_score = experiment_results
    print(f"cos_defence:{config['COS_DEFENCE']}, mean_poison_class_acc:{mean_poison_class_acc:.5f}, mean_avg_acc: {mean_avg_acc:.5f}")
    print(f"mean_poison_class_f1_score : {mean_poison_class_f1_score:.5f}, mean_avg_f1_score : {mean_avg_f1_score:.5f}")

def main():
    global base_path
    config_file = base_path + '/configs/' + sys.argv[1]
    with open(config_file) as cfg_file:
        config = yaml.safe_load(cfg_file)
        ## don't modify config here, just update the setting in corresponding modified config file and give the file name
        ## since we need to always compare cos_defence on and off, adding that variation here.
        # config['CREATE_DATASET'] = True
        config['COS_DEFENCE'] = False
        print_results(config, start_fl(config))

        config['CREATE_DATASET'] = False
        config['COS_DEFENCE'] = True
        print_results(config, start_fl(config))


if __name__ == "__main__":
    main()
