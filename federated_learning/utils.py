from sklearn.cluster import KMeans
import numpy as np

from collections import defaultdict


def cluster_params(param_arr, sep):
    params = param_arr.shape[0]
    param_ids = np.zeros((params,), dtype=int)
    count = 0
    for idx in range(params):
        kmeans = KMeans(n_clusters=2, random_state=0).fit(param_arr[idx].reshape(-1, 1))
        centers = kmeans.cluster_centers_
        param_ids[idx] = abs(centers[0] - centers[1]) >= sep
        if param_ids[idx]:
            # print(centers)
            count += 1
            print(f"{idx} : {count}")

    return param_ids, count

def find_indicative_grads(grad_bank, dataset_sel):
    ind_grad_dict = dict()
    count = 0

    # 0.1, 0.2 good for mnist
    if dataset_sel == "mnist":
        cluster_sep = 0.1
    else:
        ## for cifar10 change this
        cluster_sep = 0.1

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
