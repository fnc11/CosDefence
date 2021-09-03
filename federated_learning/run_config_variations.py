import sys
from pathlib import Path
import time
import json
import os
import yaml
import numpy as np
import copy

from FL_basic import start_fl


global base_path
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


## this method just runs the experiment k times and summarizes the results with mean and std
def run_and_summarize(config, times):
    mean_attack_srates = np.zeros(times, dtype=float)
    mean_poison_class_accs = np.zeros(times, dtype=float)
    total_class_accs_end = np.zeros(times, dtype=float)
    poison_class_accs_end = np.zeros(times, dtype=float)
    attack_srates_end = np.zeros(times, dtype=float)
    for i in range(times):
        attack_srates, source_class_accs, total_accs, mean_attack_srate, mean_poison_class_acc = start_fl(config)
        mean_attack_srates[i] = mean_attack_srate
        mean_poison_class_accs[i] = mean_poison_class_acc
        total_class_accs_end[i] = total_accs[-1]
        poison_class_accs_end[i] = source_class_accs[-1]
        attack_srates_end[i] = attack_srates[-1]
    
    print("Selected config")
    print(config)
    print(f"mean and std values after {times} random experiments")
    print(f"mean_mean_attack_srates: {np.mean(mean_attack_srates)} +- {np.std(mean_attack_srates)}")
    print(f"mean_mean_poison_class_accs: {np.mean(mean_poison_class_accs)} +- {np.std(mean_poison_class_accs)}")
    print(f"mean_total_class_accs_end: {np.mean(total_class_accs_end)} +- {np.std(total_class_accs_end)}")
    print(f"mean_poison_class_accs_end: {np.mean(poison_class_accs_end)} +- {np.std(poison_class_accs_end)}")
    print(f"mean_attack_states_end: {np.mean(attack_srates_end)} +- {np.std(attack_srates_end)}")

    summary_data = {}
    summary_data['config'] = copy.deepcopy(config)
    summary_data['mean_mean_attack_srates'] = np.mean(mean_attack_srates)
    summary_data['std_mean_attack_srates'] = np.std(mean_attack_srates)

    summary_data['mean_mean_poison_class_accs'] = np.mean(mean_poison_class_accs)
    summary_data['std_mean_poison_class_accs'] = np.std(mean_poison_class_accs)

    summary_data['mean_total_class_accs_end'] = np.mean(total_class_accs_end)
    summary_data['std_total_class_accs_end'] = np.std(total_class_accs_end)
    
    summary_data['mean_poison_class_accs_end'] = np.mean(poison_class_accs_end)
    summary_data['std_poison_class_accs_end'] = np.std(poison_class_accs_end)

    summary_data['mean_attack_srates_end'] = np.mean(attack_srates_end)
    summary_data['std_attack_srates_end'] = np.std(attack_srates_end)

    return summary_data


def main():
    global base_path
    config_file = base_path + '/configs/' + sys.argv[1]
    summary_data_list = list()
    with open(config_file) as cfg_file:
        config = yaml.safe_load(cfg_file)
        
        ## any type of variations can be added in nested structure
        ## first one without cos_defence on with fixed environment
        config['RANDOM'] = True
        config['CLIENT_FRAC'] = 0.1
        config['POISON_FRAC'] = 0.1
        config['CREATE_DATASET'] = True

        repeat = 5
        config['COS_DEFENCE'] = False
        summary_data_list.append(run_and_summarize(config, repeat))
        ## now after turning cos_defence on
        config['COS_DEFENCE'] = True
        sep_list = [0.01, 0.1, 0.5, 1.0, 8.0]
        for c_sep in sep_list:
            config['CLUSTER_SEP'] = c_sep
            config['FEATURE_FINDING_ALGO'] = 'auror'
            summary_data_list.append(run_and_summarize(config, repeat))
            config['FEATURE_FINDING_ALGO'] = 'auror_plus'
            summary_data_list.append(run_and_summarize(config, repeat))
            



        ## storing results in a json file
        json_folder = os.path.join(base_path, 'results/json_files/')
        Path(json_folder).mkdir(parents=True, exist_ok=True)
        config_details = f"{config['DATASET']}_C{config['CLIENT_FRAC']}_FDRS{config['FED_ROUNDS']}"
        file_name = 'summary_{}_{}.json'.format(config_details, {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())})
        all_summary_data = dict()
        all_summary_data['summary'] = summary_data_list
        with open(os.path.join(json_folder ,file_name), 'w') as result_file:
            json.dump(all_summary_data, result_file)




if __name__ == "__main__":
    main()
