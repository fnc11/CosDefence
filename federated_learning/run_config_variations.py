import sys
from pathlib import Path
import time
import json
import os
import yaml
import numpy as np
import copy
import itertools
import functools
from multiprocessing import Process, Pipe

from FL_basic import start_fl


global base_path
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parallelize(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):

        def invoke_method_in_different_process(*args, **kwargs):
            conn = kwargs.pop('conn')
            result = f(*args, **kwargs)
            conn.send(result)
            conn.close()

        parent_conn, child_conn = Pipe()
        kwargs['conn'] = child_conn

        child_process = Process(
            target=invoke_method_in_different_process,
            args=args,
            kwargs=kwargs
        )
        child_process.start()
        return child_process, parent_conn
    return wrapper

## this method just runs the experiment on k random dists and summarizes the results with mean and std
# @parallelize
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


# def run_off_on_summarize(config, random_dists):
#     mean_poison_class_accs_off = np.zeros(random_dists, dtype=float)
#     mean_avg_accs_off = np.zeros(random_dists, dtype=float)
#     mean_poison_class_f1_scores_off = np.zeros(random_dists, dtype=float)
#     mean_avg_f1_scores_off = np.zeros(random_dists, dtype=float)


# @parallelize
# def run_off_on_summarize(config, random_dists):
#     ## this method first generates a dataset radomly and then fixes the process so
#     ## that we can see how cos_defence off and on works on same environment

#     mean_poison_class_accs_on = np.zeros(random_dists, dtype=float)
#     mean_avg_accs_on = np.zeros(random_dists, dtype=float)
#     mean_poison_class_f1_scores_on = np.zeros(random_dists, dtype=float)
#     mean_avg_f1_scores_on = np.zeros(random_dists, dtype=float)

#     config['RANDOM_PROCESS'] = False
#     for i in range(random_dists):
#         config['COS_DEFENCE'] = False
#         mean_poison_class_accs_off[i], mean_avg_accs_off[i], mean_poison_class_f1_scores_off[i], mean_avg_f1_scores_off[i] = start_fl(config, dist_id=i)

#         config['COS_DEFENCE'] = True
#         mean_poison_class_accs_on[i], mean_avg_accs_on[i], mean_poison_class_f1_scores_on[i], mean_avg_f1_scores_on[i] = start_fl(config, dist_id=i)


#     summary_data = {}
#     cdf_off_summary = {}
#     cdf_on_summary = {}

#     print("Selected config")
#     config['COS_DEFENCE'] = False
#     print(config)
#     print(f"mean and std values after {random_dists} random experiments when cos_defence: {config['COS_DEFENCE']}")
#     print(f"mean_mean_poison_class_accs: {np.mean(mean_poison_class_accs_off)} +- {np.std(mean_poison_class_accs_off)}")
#     print(f"mean_mean_avg_accs: {np.mean(mean_avg_accs_off)} +- {np.std(mean_avg_accs_off)}")
#     print(f"mean_mean_poison_class_f1_scores: {np.mean(mean_poison_class_f1_scores_off)} +- {np.std(mean_poison_class_f1_scores_off)}")
#     print(f"mean_mean_avg_f1_scores: {np.mean(mean_avg_f1_scores_off)} +- {np.std(mean_avg_f1_scores_off)}")
    
#     cdf_off_summary['config'] = copy.deepcopy(config)
#     cdf_off_summary['mean_mean_poison_class_accs_off'] = np.mean(mean_poison_class_accs_off)
#     cdf_off_summary['std_mean_poison_class_accs_off'] = np.std(mean_poison_class_accs_off)

#     cdf_off_summary['mean_mean_avg_accs_off'] = np.mean(mean_avg_accs_off)
#     cdf_off_summary['std_mean_avg_accs_off'] = np.std(mean_avg_accs_off)
    
#     cdf_off_summary['mean_mean_poison_class_f1_scores_off'] = np.mean(mean_poison_class_f1_scores_off)
#     cdf_off_summary['std_mean_poison_class_f1_scores_off'] = np.std(mean_poison_class_f1_scores_off)

#     cdf_off_summary['mean_mean_avg_f1_scores_off'] = np.mean(mean_avg_f1_scores_off)
#     cdf_off_summary['std_mean_avg_f1_scores_off'] = np.std(mean_avg_f1_scores_off)
#     summary_data['cdf_off_summary'] = cdf_off_summary


    
#     print("Selected config")
#     config['COS_DEFENCE'] = True
#     print(config)
#     print(f"mean and std values after {random_dists} random experiments when cos_defence: {config['COS_DEFENCE']}")
#     print(f"mean_mean_poison_class_accs: {np.mean(mean_poison_class_accs_on)} +- {np.std(mean_poison_class_accs_on)}")
#     print(f"mean_mean_avg_accs: {np.mean(mean_avg_accs_on)} +- {np.std(mean_avg_accs_on)}")
#     print(f"mean_mean_poison_class_f1_scores: {np.mean(mean_poison_class_f1_scores_on)} +- {np.std(mean_poison_class_f1_scores_on)}")
#     print(f"mean_mean_avg_f1_scores: {np.mean(mean_avg_f1_scores_on)} +- {np.std(mean_avg_f1_scores_on)}")
    
#     cdf_on_summary['config'] = copy.deepcopy(config)
#     cdf_on_summary['mean_mean_poison_class_accs_on'] = np.mean(mean_poison_class_accs_on)
#     cdf_on_summary['std_mean_poison_class_accs_on'] = np.std(mean_poison_class_accs_on)

#     cdf_on_summary['mean_mean_avg_accs_on'] = np.mean(mean_avg_accs_on)
#     cdf_on_summary['std_mean_avg_accs_on'] = np.std(mean_avg_accs_on)
    
#     cdf_on_summary['mean_mean_poison_class_f1_scores_on'] = np.mean(mean_poison_class_f1_scores_on)
#     cdf_on_summary['std_mean_poison_class_f1_scores_on'] = np.std(mean_poison_class_f1_scores_on)

#     cdf_on_summary['mean_mean_avg_f1_scores_on'] = np.mean(mean_avg_f1_scores_on)
#     cdf_on_summary['std_mean_avg_f1_scores_on'] = np.std(mean_avg_f1_scores_on)
#     summary_data['cdf_on_summary'] = cdf_on_summary
#     return summary_data


def run_with_parallization(summary_data_list, initial_config, random_dists):
    configs_to_go_over = {
        'CLIENT_FRAC': [0.2],
        'POISON_FRAC': [0.0, 0.1, 0.2, 0.4, 0.5]
    }
    child_processes = []
    for config_values in itertools.product(*configs_to_go_over.values()):
        for setting, value in zip(configs_to_go_over.keys(), config_values):
            initial_config[setting] = value
        child_processes.append(run_and_summarize(config=initial_config, random_dists=random_dists))
    for process, parent_conn in child_processes:
        summary_data_list.append(parent_conn.recv())
        process.join()

    return summary_data_list
    


def run_sequentially(summary_data_list, initial_config, random_dists):
    initial_config['COS_DEFENCE'] = True
    initial_config['CLIENT_FRAC'] = 0.2
    p_fracs = [0.1, 0.2, 0.3]
    for p_frac in p_fracs:
        initial_config['POISON_FRAC'] = p_frac
        summary_data_list.append(run_and_summarize(initial_config, random_dists))
    
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
        # random_dists = 2
                
        ## any type of variations can be added in nested structure
        ## first one without cos_defence on with fixed environment
        # summary_data_list = run_with_parallization(summary_data_list, config, 5)
        summary_data_list = run_sequentially(summary_data_list, config, 5)



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
