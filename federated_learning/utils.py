from torch import tensor
from sklearn.cluster import KMeans
import warnings
import logging
import copy

import numpy as np
import json

from collections import defaultdict


def cluster_params(param_arr, feature_finding_algo, sep):
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
                if feature_finding_algo == "auror":
                    logging.debug(centers)
                    node_count += 1
                    logging.debug(f"{idx} : {node_count}")
                elif feature_finding_algo == "auror_plus":
                    labels = kmeans.predict(param_arr[idx].reshape(-1, 1))
                    _vals, counts = np.unique(labels, return_counts=True)
                    mins_list.append(np.min(counts))

        ## we go one step further, and filter out more nodes
        if feature_finding_algo == "auror_plus":
            if len(mins_list) > 1:   
                mins_arr = np.array(mins_list).reshape(-1, 1)
                min_kmeans = KMeans(n_clusters=2, random_state=0).fit(mins_arr)
                min_centers = min_kmeans.cluster_centers_
                logging.debug(min_centers)
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
            else:
                ## this takes care the case where only one neural unit was found important in
                ## that layer
                for idx in range(params):
                    if param_ids[idx]:
                        node_count += 1

    return param_ids, node_count

def filter_big():
    pass

def find_indicative_grads(grad_bank, feature_finding_algo="auror", cluster_sep=0.1):
    ind_grad_dict = dict()
    count = 0

    for layer_name in grad_bank.keys():
        logging.info(f"Finding features in {layer_name}")
        tensor_shape = grad_bank[layer_name][0].shape
        logging.info(tensor_shape)
        if feature_finding_algo == "none":
            ## all neural units are important
            feature_arr = np.ones(tensor_shape)
            icount = feature_arr.size
        else:
            logging.info("Number of models in bank")
            logging.info(len(grad_bank[layer_name]))
            diff_model_params = [arr.detach().clone().flatten().cpu() for arr in grad_bank[layer_name]]
            # print(diff_model_params)
            param_arr = np.stack(diff_model_params, axis=0).transpose()
            ## now this param_arr contains each row of values from different models, so we just cluster all row values
            feature_arr, icount = cluster_params(param_arr, feature_finding_algo, cluster_sep)
            # print(feature_arr)
            feature_arr = feature_arr.reshape(tensor_shape)
            # print(feature_arr)
        ind_grad_dict[layer_name] = feature_arr
        count += icount

        ## to save the important feature connections in layers
        # with open(f'{layer_name}.npy', 'wb') as f:
        #     np.save(f, feature_arr)
    
    return ind_grad_dict, count


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)



class ComputationRecord:
    """[summary]
    This class is for storing all variables we need to keep track of during computation.
    """
    def __init__(self, config):
        ## experiment config
        self.config = copy.deepcopy(config)
        ## system trust vector, determines trustworthiness of clients in the setup
        self.csystem_tvec = None
        ## system trust matrix, how much clients trust each other
        self.csystem_tmat = None

        ## global grads, stores aggregated values of the grads, we will initialize it based on the number of parameters it stores
        ## e.g. last layer of the model
        self.sel_layer_names = None
        self.agg_grads = None
        self.initial_agg_grads = None

        ## grad_bank is a dictionary which stores gradients, when Auror algo is used to find important neural units in the model,
        ## it store gradients of all client models selected in first K federated rounds layer wise
        self.grad_bank = None
        ## boolean flag that indicates until when grad should be saved for finding important units
        self.save_for_ft_finding = None
        ## stores the location of important gradients layerwise, once we have the result from Auror algo
        self.indicative_grads = None
        ## boolean flag, when to start collecting grads from important neural units
        self.collect_features = None

        ## for analytics, we plot what trust value clients got, all list variables
        ## in case of collab mode validation client ids are not stored
        self.client_ids = None
        self.vclient_ids = None 
        self.all_trust_vals = None
        self.all_client_types = None