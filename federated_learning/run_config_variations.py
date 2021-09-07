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


def run_off_on_summarize(config, times):
    ## this method first generates a dataset radomly and then fixes the process so
    ## that we can see how cos_defence off and on works on same environment
    mean_attack_srates_off = np.zeros(times, dtype=float)
    mean_poison_class_accs_off = np.zeros(times, dtype=float)
    total_class_accs_end_off = np.zeros(times, dtype=float)
    poison_class_accs_end_off = np.zeros(times, dtype=float)
    attack_srates_end_off = np.zeros(times, dtype=float)

    mean_attack_srates_on = np.zeros(times, dtype=float)
    mean_poison_class_accs_on = np.zeros(times, dtype=float)
    total_class_accs_end_on = np.zeros(times, dtype=float)
    poison_class_accs_end_on = np.zeros(times, dtype=float)
    attack_srates_end_on = np.zeros(times, dtype=float)

    config['RANDOM_PROCESS'] = False
    for i in range(times):
        config['CREATE_DATASET'] = True
        config['COS_DEFENCE'] = False

        attack_srates_off, source_class_accs_off, total_accs_off, mean_attack_srate_off, mean_poison_class_acc_off = start_fl(config)
        mean_attack_srates_off[i] = mean_attack_srate_off
        mean_poison_class_accs_off[i] = mean_poison_class_acc_off
        total_class_accs_end_off[i] = total_accs_off[-1]
        poison_class_accs_end_off[i] = source_class_accs_off[-1]
        attack_srates_end_off[i] = attack_srates_off[-1]

        config['CREATE_DATASET'] = False
        config['COS_DEFENCE'] = True

        attack_srates_on, source_class_accs_on, total_accs_on, mean_attack_srate_on, mean_poison_class_acc_on = start_fl(config)
        mean_attack_srates_on[i] = mean_attack_srate_on
        mean_poison_class_accs_on[i] = mean_poison_class_acc_on
        total_class_accs_end_on[i] = total_accs_on[-1]
        poison_class_accs_end_on[i] = source_class_accs_on[-1]
        attack_srates_end_on[i] = attack_srates_on[-1]


    summary_data = {}
    cdf_off_summary = {}
    cdf_on_summary = {}

    print("Selected config")
    config['COS_DEFENCE'] = False
    print(config)
    print(f"mean and std values after {times} random experiments when cos_defence: {config['COS_DEFENCE']}")
    print(f"mean_mean_attack_srates: {np.mean(mean_attack_srates_off)} +- {np.std(mean_attack_srates_off)}")
    print(f"mean_mean_poison_class_accs: {np.mean(mean_poison_class_accs_off)} +- {np.std(mean_poison_class_accs_off)}")
    print(f"mean_total_class_accs_end: {np.mean(total_class_accs_end_off)} +- {np.std(total_class_accs_end_off)}")
    print(f"mean_poison_class_accs_end: {np.mean(poison_class_accs_end_off)} +- {np.std(poison_class_accs_end_off)}")
    print(f"mean_attack_states_end: {np.mean(attack_srates_end_off)} +- {np.std(attack_srates_end_off)}")

    cdf_off_summary['config'] = copy.deepcopy(config)
    cdf_off_summary['mean_mean_attack_srates_off'] = np.mean(mean_attack_srates_off)
    cdf_off_summary['std_mean_attack_srates_off'] = np.std(mean_attack_srates_off)

    cdf_off_summary['mean_mean_poison_class_accs_off'] = np.mean(mean_poison_class_accs_off)
    cdf_off_summary['std_mean_poison_class_accs_off'] = np.std(mean_poison_class_accs_off)

    cdf_off_summary['mean_total_class_accs_end_off'] = np.mean(total_class_accs_end_off)
    cdf_off_summary['std_total_class_accs_end_off'] = np.std(total_class_accs_end_off)
    
    cdf_off_summary['mean_poison_class_accs_end_off'] = np.mean(poison_class_accs_end_off)
    cdf_off_summary['std_poison_class_accs_end_off'] = np.std(poison_class_accs_end_off)

    cdf_off_summary['mean_attack_srates_end_off'] = np.mean(attack_srates_end_off)
    cdf_off_summary['std_attack_srates_end_off'] = np.std(attack_srates_end_off)
    summary_data['cdf_off_summary'] = cdf_off_summary
    
    print("Selected config")
    config['COS_DEFENCE'] = True
    print(config)
    print(f"mean and std values after {times} random experiments when cos_defence: {config['COS_DEFENCE']}")
    print(f"mean_mean_attack_srates: {np.mean(mean_attack_srates_on)} +- {np.std(mean_attack_srates_on)}")
    print(f"mean_mean_poison_class_accs: {np.mean(mean_poison_class_accs_on)} +- {np.std(mean_poison_class_accs_on)}")
    print(f"mean_total_class_accs_end: {np.mean(total_class_accs_end_on)} +- {np.std(total_class_accs_end_on)}")
    print(f"mean_poison_class_accs_end: {np.mean(poison_class_accs_end_on)} +- {np.std(poison_class_accs_end_on)}")
    print(f"mean_attack_states_end: {np.mean(attack_srates_end_on)} +- {np.std(attack_srates_end_on)}")



    cdf_on_summary['config'] = copy.deepcopy(config)
    cdf_on_summary['mean_mean_attack_srates_on'] = np.mean(mean_attack_srates_on)
    cdf_on_summary['std_mean_attack_srates_on'] = np.std(mean_attack_srates_on)

    cdf_on_summary['mean_mean_poison_class_accs_on'] = np.mean(mean_poison_class_accs_on)
    cdf_on_summary['std_mean_poison_class_accs_on'] = np.std(mean_poison_class_accs_on)

    cdf_on_summary['mean_total_class_accs_end_on'] = np.mean(total_class_accs_end_on)
    cdf_on_summary['std_total_class_accs_end_on'] = np.std(total_class_accs_end_on)
    
    cdf_on_summary['mean_poison_class_accs_end_on'] = np.mean(poison_class_accs_end_on)
    cdf_on_summary['std_poison_class_accs_end_on'] = np.std(poison_class_accs_end_on)

    cdf_on_summary['mean_attack_srates_end_on'] = np.mean(attack_srates_end_on)
    cdf_on_summary['std_attack_srates_end_on'] = np.std(attack_srates_end_on)
    summary_data['cdf_on_summary'] = cdf_on_summary

    return summary_data

def main():
    global base_path
    config_file = base_path + '/configs/' + sys.argv[1]
    summary_data_list = list()
    with open(config_file) as cfg_file:
        config = yaml.safe_load(cfg_file)
        
        ## any type of variations can be added in nested structure
        ## first one without cos_defence on with fixed environment
        config['CLIENT_FRAC'] = 0.1
        config['POISON_FRAC'] = 0.0

        repeat = 10
        summary_data_list.append(run_off_on_summarize(config, repeat))
        
        config['POISON_FRAC'] = 0.1
        summary_data_list.append(run_off_on_summarize(config, repeat))

        # repeat = 5
        # config['COS_DEFENCE'] = False
        # summary_data_list.append(run_and_summarize(config, repeat))
        # ## now after turning cos_defence on
        # sep_list = [0.001, 0.01, 0.1, 1.0, 2.0]
        # config['COS_DEFENCE'] = True
        # for c_sep in sep_list:
        #     config['CLUSTER_SEP'] = c_sep
        #     summary_data_list.append(run_and_summarize(config, repeat))

        # config['POISON_FRAC'] = 0.1
        # config['COS_DEFENCE'] = False
        # summary_data_list.append(run_and_summarize(config, repeat))
        # ## now after turning cos_defence on
        # config['COS_DEFENCE'] = True
        # for c_sep in sep_list:
        #     config['CLUSTER_SEP'] = c_sep
        #     summary_data_list.append(run_and_summarize(config, repeat))

        # config['POISON_FRAC'] = 0.2
        # config['COS_DEFENCE'] = False
        # summary_data_list.append(run_and_summarize(config, repeat))
        # ## now after turning cos_defence on
        # config['COS_DEFENCE'] = True
        # for c_sep in sep_list:
        #     config['CLUSTER_SEP'] = c_sep
        #     summary_data_list.append(run_and_summarize(config, repeat))

        # config['POISON_FRAC'] = 0.3
        # config['COS_DEFENCE'] = False
        # summary_data_list.append(run_and_summarize(config, repeat))
        # ## now after turning cos_defence on
        # config['COS_DEFENCE'] = True
        # for c_sep in sep_list:
        #     config['CLUSTER_SEP'] = c_sep
        #     summary_data_list.append(run_and_summarize(config, repeat))

        # config['POISON_FRAC'] = 0.4
        # config['COS_DEFENCE'] = False
        # summary_data_list.append(run_and_summarize(config, repeat))
        # ## now after turning cos_defence on
        # config['COS_DEFENCE'] = True
        # for c_sep in sep_list:
        #     config['CLUSTER_SEP'] = c_sep
        #     summary_data_list.append(run_and_summarize(config, repeat))
        # sep_list = [0.001, 0.002, 0.005, 0.007, 0.01]
        # config['FEATURE_FINDING_ALGO'] = 'auror'

        # config['CONSIDER_LAYERS'] = 'l1'
        # for c_sep in sep_list:
        #     config['CLUSTER_SEP'] = c_sep
        #     summary_data_list.append(run_and_summarize(config, repeat))

        # config['CONSIDER_LAYERS'] = 'l2'
        # for c_sep in sep_list:
        #     config['CLUSTER_SEP'] = c_sep
        #     summary_data_list.append(run_and_summarize(config, repeat))

        # config['CONSIDER_LAYERS'] = 'f1l1'
        # for c_sep in sep_list:
        #     config['CLUSTER_SEP'] = c_sep
        #     summary_data_list.append(run_and_summarize(config, repeat))
            
 


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
