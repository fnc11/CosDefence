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

## this method just runs the experiment on k random dists and summarizes the results with mean and std
def run_and_summarize(config, random_dists):
    mean_poison_class_accs = np.zeros(random_dists, dtype=float)
    mean_avg_accs = np.zeros(random_dists, dtype=float)
    mean_poison_class_f1_scores = np.zeros(random_dists, dtype=float)
    mean_avg_f1_scores = np.zeros(random_dists, dtype=float)
    for i in range(random_dists):
        mean_poison_class_accs[i], mean_avg_accs[i], mean_poison_class_f1_scores[i], mean_avg_f1_scores[i] = start_fl(config, dist_id=i)

    print("Selected config")
    print(config)
    print(f"mean and std values after {random_dists} random experiments")
    print(f"mean_mean_poison_class_accs: {np.mean(mean_poison_class_accs)} +- {np.std(mean_poison_class_accs)}")
    print(f"mean_mean_avg_accs: {np.mean(mean_avg_accs)} +- {np.std(mean_avg_accs)}")
    print(f"mean_mean_poison_class_f1_scores: {np.mean(mean_poison_class_f1_scores)} +- {np.std(mean_poison_class_f1_scores)}")
    print(f"mean_mean_avg_f1_scores: {np.mean(mean_avg_f1_scores)} +- {np.std(mean_avg_f1_scores)}")

    summary_data = {}
    summary_data['config'] = copy.deepcopy(config)

    summary_data['mean_mean_poison_class_accs'] = np.mean(mean_poison_class_accs)
    summary_data['std_mean_poison_class_accs'] = np.std(mean_poison_class_accs)

    summary_data['mean_mean_avg_accs'] = np.mean(mean_avg_accs)
    summary_data['std_mean_avg_accs'] = np.std(mean_avg_accs)
    
    summary_data['mean_mean_poison_class_f1_scores'] = np.mean(mean_poison_class_f1_scores)
    summary_data['std_mean_poison_class_f1_scores'] = np.std(mean_poison_class_f1_scores)

    summary_data['mean_mean_avg_f1_scores'] = np.mean(mean_avg_f1_scores)
    summary_data['std_mean_avg_f1_scores'] = np.std(mean_avg_f1_scores)

    return summary_data


def run_sequentially(summary_data_list, config, random_dists):
    config['COS_DEFENCE'] = True
    pardoning_factors = [0.0, 0.2, 0.8, 1.0, 1.2, 2.0]
    for factor in pardoning_factors:
        config['HONEST_PARDON_FACTOR'] = factor
        try:
            summary_data_list.append(run_and_summarize(config, random_dists))
        except Exception as e:
            print(e)
    
    return summary_data_list




def main():
    global base_path
    config_file = base_path + '/configs/' + sys.argv[1]
    summary_data_list = list()
    with open(config_file) as cfg_file:
        config = yaml.safe_load(cfg_file)

        ## First create 5, 10 dists which you want to try, then run for those distributions
        ## distribution creation can be done in single time using prepare_data script
        ## we don't create these dist again, because it increases randomness and we can't compare
        ## the results with different settings.
        random_dists = 5
        start_time = time.perf_counter()
        summary_data_list = run_sequentially(summary_data_list, config, random_dists)
        end_time = time.perf_counter()
        print(f"Took {end_time-start_time} secs")



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
