from sklearn.cluster import KMeans
import warnings

import numpy as np

from collections import defaultdict


def cluster_params(param_arr, sep):
    params = param_arr.shape[0]
    param_ids = np.zeros((params,), dtype=int)
    node_count = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mins_list = list()
        for idx in range(params):
            kmeans = KMeans(n_clusters=2, random_state=0).fit(param_arr[idx].reshape(-1, 1))
            centers = kmeans.cluster_centers_
            param_ids[idx] = abs(centers[0] - centers[1]) >= sep
            if param_ids[idx]:
                # print(centers)
                # count += 1
                # print(f"{idx} : {count}")
                labels = kmeans.predict(param_arr[idx].reshape(-1, 1))
                _vals, counts = np.unique(labels, return_counts=True)
                mins_list.append(np.min(counts))

        if len(mins_list) > 1:   
            mins_arr = np.array(mins_list).reshape(-1, 1)
            min_kmeans = KMeans(n_clusters=2, random_state=0).fit(mins_arr)
            min_centers = min_kmeans.cluster_centers_
            print(min_centers)
            min_labels = min_kmeans.predict(mins_arr)

            ## cluster with less numbers
            min_idx = np.argmin(min_centers)
            
            ## select only those nodes which are changed by less clients
            k = 0
            for idx in range(params):
                if param_ids[idx]:
                    param_ids[idx] = (min_labels[k] == min_idx)
                    if param_ids[idx]:
                        node_count += 1
                    k += 1

    return param_ids, node_count

def filter_big():
    pass

def find_indicative_grads(grad_bank, dataset_sel, cluster_sep=0.1):
    ind_grad_dict = dict()
    count = 0

    # # 0.1, 0.2 good for mnist
    # if dataset_sel == "mnist":
    #     cluster_sep = 0.1
    # else:
    #     ## for cifar10 change this
    #     cluster_sep = 0.5

    for layer_name in grad_bank.keys():
        
        tensor_shape = grad_bank[layer_name][0].shape
        print(f"Finding features in {layer_name}")
        print(tensor_shape)
        print("Number of models in bank")
        print(len(grad_bank[layer_name]))

        diff_model_params = [arr.detach().clone().flatten().cpu() for arr in grad_bank[layer_name]]
        # print(diff_model_params)
        param_arr = np.stack(diff_model_params, axis=0).transpose()
        ## now this param_arr contains each row of values from different models, so we just cluster all row values
        feature_arr, icount = cluster_params(param_arr, cluster_sep)
        count += icount
        # print(feature_arr)
        feature_arr = feature_arr.reshape(tensor_shape)
        # print(feature_arr)
        ind_grad_dict[layer_name] = feature_arr
        with open(f'{layer_name}.npy', 'wb') as f:
            np.save(f, feature_arr)
    
    return ind_grad_dict, count
