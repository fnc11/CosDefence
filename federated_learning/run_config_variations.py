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


def run_off_on_summarize(config, random_dists):
    mean_poison_class_accs_off = np.zeros(random_dists, dtype=float)
    mean_avg_accs_off = np.zeros(random_dists, dtype=float)
    mean_poison_class_f1_scores_off = np.zeros(random_dists, dtype=float)
    mean_avg_f1_scores_off = np.zeros(random_dists, dtype=float)

    mean_poison_class_accs_on = np.zeros(random_dists, dtype=float)
    mean_avg_accs_on = np.zeros(random_dists, dtype=float)
    mean_poison_class_f1_scores_on = np.zeros(random_dists, dtype=float)
    mean_avg_f1_scores_on = np.zeros(random_dists, dtype=float)

    config['RANDOM_PROCESS'] = False
    for i in range(random_dists):
        config['COS_DEFENCE'] = False
        mean_poison_class_accs_off[i], mean_avg_accs_off[i], mean_poison_class_f1_scores_off[i], mean_avg_f1_scores_off[i] = start_fl(config, dist_id=i)

        config['COS_DEFENCE'] = True
        mean_poison_class_accs_on[i], mean_avg_accs_on[i], mean_poison_class_f1_scores_on[i], mean_avg_f1_scores_on[i] = start_fl(config, dist_id=i)


    summary_data = {}
    cdf_off_summary = {}
    cdf_on_summary = {}

    print("Selected config")
    config['COS_DEFENCE'] = False
    print(config)
    print(f"mean and std values after {random_dists} random experiments when cos_defence: {config['COS_DEFENCE']}")
    print(f"mean_mean_poison_class_accs: {np.mean(mean_poison_class_accs_off)} +- {np.std(mean_poison_class_accs_off)}")
    print(f"mean_mean_avg_accs: {np.mean(mean_avg_accs_off)} +- {np.std(mean_avg_accs_off)}")
    print(f"mean_mean_poison_class_f1_scores: {np.mean(mean_poison_class_f1_scores_off)} +- {np.std(mean_poison_class_f1_scores_off)}")
    print(f"mean_mean_avg_f1_scores: {np.mean(mean_avg_f1_scores_off)} +- {np.std(mean_avg_f1_scores_off)}")
    
    cdf_off_summary['config'] = copy.deepcopy(config)
    cdf_off_summary['mean_mean_poison_class_accs_off'] = np.mean(mean_poison_class_accs_off)
    cdf_off_summary['std_mean_poison_class_accs_off'] = np.std(mean_poison_class_accs_off)

    cdf_off_summary['mean_mean_avg_accs_off'] = np.mean(mean_avg_accs_off)
    cdf_off_summary['std_mean_avg_accs_off'] = np.std(mean_avg_accs_off)
    
    cdf_off_summary['mean_mean_poison_class_f1_scores_off'] = np.mean(mean_poison_class_f1_scores_off)
    cdf_off_summary['std_mean_poison_class_f1_scores_off'] = np.std(mean_poison_class_f1_scores_off)

    cdf_off_summary['mean_mean_avg_f1_scores_off'] = np.mean(mean_avg_f1_scores_off)
    cdf_off_summary['std_mean_avg_f1_scores_off'] = np.std(mean_avg_f1_scores_off)
    summary_data['cdf_off_summary'] = cdf_off_summary


    
    print("Selected config")
    config['COS_DEFENCE'] = True
    print(config)
    print(f"mean and std values after {random_dists} random experiments when cos_defence: {config['COS_DEFENCE']}")
    print(f"mean_mean_poison_class_accs: {np.mean(mean_poison_class_accs_on)} +- {np.std(mean_poison_class_accs_on)}")
    print(f"mean_mean_avg_accs: {np.mean(mean_avg_accs_on)} +- {np.std(mean_avg_accs_on)}")
    print(f"mean_mean_poison_class_f1_scores: {np.mean(mean_poison_class_f1_scores_on)} +- {np.std(mean_poison_class_f1_scores_on)}")
    print(f"mean_mean_avg_f1_scores: {np.mean(mean_avg_f1_scores_on)} +- {np.std(mean_avg_f1_scores_on)}")
    
    cdf_on_summary['config'] = copy.deepcopy(config)
    cdf_on_summary['mean_mean_poison_class_accs_on'] = np.mean(mean_poison_class_accs_on)
    cdf_on_summary['std_mean_poison_class_accs_on'] = np.std(mean_poison_class_accs_on)

    cdf_on_summary['mean_mean_avg_accs_on'] = np.mean(mean_avg_accs_on)
    cdf_on_summary['std_mean_avg_accs_on'] = np.std(mean_avg_accs_on)
    
    cdf_on_summary['mean_mean_poison_class_f1_scores_on'] = np.mean(mean_poison_class_f1_scores_on)
    cdf_on_summary['std_mean_poison_class_f1_scores_on'] = np.std(mean_poison_class_f1_scores_on)

    cdf_on_summary['mean_mean_avg_f1_scores_on'] = np.mean(mean_avg_f1_scores_on)
    cdf_on_summary['std_mean_avg_f1_scores_on'] = np.std(mean_avg_f1_scores_on)
    summary_data['cdf_on_summary'] = cdf_on_summary
    return summary_data

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
        random_dists = 2
                
        ## any type of variations can be added in nested structure
        ## first one without cos_defence on with fixed environment
        config['CLIENT_FRAC'] = 0.2
        
        p_fracs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        for p_frac in p_fracs:
            config['POISON_FRAC'] = p_frac
            config['COS_DEFENCE'] = False
            summary_data_list.append(run_and_summarize(config, random_dists))
            config['COS_DEFENCE'] = True
            summary_data_list.append(run_and_summarize(config, random_dists))

        # config['POISON_FRAC'] = 0.0
        # summary_data_list.append(run_off_on_summarize(config, repeat))
        
        # config['POISON_FRAC'] = 0.1
        # summary_data_list.append(run_off_on_summarize(config, repeat))

        # config['POISON_FRAC'] = 0.2
        # summary_data_list.append(run_off_on_summarize(config, repeat))

        # config['POISON_FRAC'] = 0.3
        # summary_data_list.append(run_off_on_summarize(config, repeat))

        # config['POISON_FRAC'] = 0.4
        # summary_data_list.append(run_off_on_summarize(config, repeat))

        # config['POISON_FRAC'] = 0.5
        # summary_data_list.append(run_off_on_summarize(config, repeat))

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
