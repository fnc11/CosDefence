import copy
from utils import find_indicative_grads
import time
import os
import logging
import numpy as np
import pandas as pd
from numpy.random import default_rng
from pathlib import Path


from prepare_data import create_client_data, create_client_data_loaders, get_test_data_loader
from available_models import get_model, get_selected_layers
import json
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

import torch
import torch.nn as nn
import torch.optim as optim


## project path
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
## experiment config
config = None

## system trust vector, determines trustworthiness of clients in the setup
current_system_trust_vec = None

## system trust matrix, how much clients trust each other
current_system_trust_mat = None

## global grads, stores aggregated values of the grads, we will initialize it based on the number of parameters it stores
## e.g. last layer of the model
sel_layer_names = None
aggregate_grads = None
initial_aggregate_grads = None

## grad_bank is a dictionary which stores gradients, when Auror algo is used to find important neural units in the model,
## it store gradients of all client models selected in first K federated rounds layer wise
grad_bank = None
## boolean flag that indicates until when grad should be saved for finding important units
save_for_feature_finding = None
## stores the location of important gradients layerwise, once we have the result from Auror algo
indicative_grads = None
## boolean flag, when to start collecting grads from important neural units
collect_features = None

## for analytics, we plot what trust value clients got, all list variables
## in case of collab mode validation client ids are not stored
client_ids = None
validation_client_ids = None 
all_trust_vals = None
all_client_types = None



def set_initial_trust_vec(dist_type="ones"):
    if dist_type == "ones":
        trust_vec = np.ones((100), dtype=float)
    elif dist_type == "zeros":
        trust_vec = np.zeros((100), dtype=float)
    elif dist_type == "random":
        trust_vec = np.random.random(100)
    elif dist_type == "uniform":
        trust_vec = np.random.uniform(0.0, 1.0, 100)
    else:
        mu, sigma = 0.5, 0.1 # mean and standard deviation
        trust_vec = np.random.normal(mu, sigma, 100)

    if trust_vec.sum() > 0.0:
        trust_vec /= trust_vec.sum()
    logging.info("Initial Trust vector set by method")
    logging.info(trust_vec)
    return trust_vec

def set_initial_trust_mat(dist_type="ones"):
    normalized_local_trusts = list()
    if dist_type == "ones":
        for _ in range(100):
            local_trust = np.ones(100, dtype=float)
            normalized_local_trusts.append(local_trust/local_trust.sum())
    elif dist_type == "random":
        for _ in range(100):
            local_trust = np.random.random(100)
            normalized_local_trusts.append(local_trust/local_trust.sum())
    elif dist_type == "uniform":
        for _ in range(100):
            local_trust = np.random.uniform(0.0, 1.0, 100)
            normalized_local_trusts.append(local_trust/local_trust.sum())
    elif dist_type == "zeros":
        for _ in range(100):
            local_trust = np.zeros(100, dtype=float)
            normalized_local_trusts.append(local_trust)
        logging.info("Trust mat started with all zeros")
        return np.stack(normalized_local_trusts)
    else:
        ## should we use multivariate normal distribution?
        mu, sigma = 0.5, 0.1 # mean and standard deviation
        for _ in range(100):
            local_trust = np.random.normal(mu, sigma, 100)
            normalized_local_trusts.append(local_trust/local_trust.sum())

    trust_mat = np.stack(normalized_local_trusts)
    logging.info("Initial Trust Mat set by method")
    logging.info(trust_mat)
    return trust_mat


def init_validation_clients(poisoned_clients, rng):
    global config
    global current_system_trust_vec
    # selecting validation clients
    validation_clients_available = [client for client in range(config['TOTAL_CLIENTS']) if client not in poisoned_clients]
    validation_clients = rng.choice(validation_clients_available, size=int(config['GAMMA'] * config['TOTAL_CLIENTS']), replace=False)
    
    # validation client's trust distribution
    additional_trust_vector = np.zeros(config['TOTAL_CLIENTS'])
    # replaces the zeros with 1 amount of trust value given to validation client
    np.put(additional_trust_vector, validation_clients, max(0.01, config["TRUST_INC"]*np.max(current_system_trust_vec)))

    # we add this additional trust vector to the initial system trust so that system has more trust on validation clients
    # regardless of the trust distribution we use during initialization 
    logging.info("Before trust addition")
    logging.info(current_system_trust_vec.shape)
    logging.info(additional_trust_vector.shape)

    current_system_trust_vec += additional_trust_vector
    current_system_trust_vec /= current_system_trust_vec.sum()

    logging.info("After Validation client trust addition")
    logging.info(current_system_trust_vec)

    return validation_clients

def eigen_trust(alpha=0.5, epsilon=0.00001):
    logging.debug("In Eigen Trust")
    ## need to check their dimensions but for now we make it so that they can be multiplied
    global current_system_trust_mat
    global current_system_trust_vec
    matrix_M = np.copy(current_system_trust_mat)

    pre_trust = np.copy(current_system_trust_vec)
    # logging.debug(pre_trust)
    
    n = 100
    new_trust = pre_trust
    # delta_diffs = list()
    for i in range(n):
        old_trust = new_trust
        logging.debug(new_trust)
        new_trust = np.matmul(matrix_M.transpose(), old_trust)
        new_trust = alpha*pre_trust + (1-alpha)*new_trust 
        
        dist = np.linalg.norm(new_trust-old_trust)
        # delta_diffs.append(dist)
        if dist <= epsilon:
            logging.debug(f"\nno change in trust values after iter: {i} with alpha: {alpha}")
            # logging.debug(new_trust)
            break

    ## normalize it to sum upto 1
    new_trust /= new_trust.sum()

    current_system_trust_vec = new_trust


def identify_poisoned(clients_selected, poisoned_clients):
    posioned_client_selected = []
    for client in poisoned_clients:
        if client in clients_selected:
            posioned_client_selected.append(client)

    return posioned_client_selected

def fill_up_rem_trust_mat_and_vec(total_clients, initial_validation_clients, fill_up=True):
    global current_system_trust_mat
    global current_system_trust_vec

    ## Since we pick clients based on trust vec while cos_defence is selected, so we need to make sure that 
    ## trust vec sums upto 1
    current_system_trust_vec /= current_system_trust_vec.sum()

    ## we first normalize the trust matrix and then find mean and std for sampling where values are zero
    sum_of_rows = current_system_trust_mat.sum(axis=1)
    current_system_trust_mat = current_system_trust_mat / sum_of_rows[:, np.newaxis]
    
    ## we also check here how many clients out of total clients were never selected while initial 
    ## trust building
    not_selected_before_starting = 0
    if fill_up:
        ## step 1, find mean, std from the non_zero values and also note down these clients
        for i in range(total_clients):
            zero_interaction_clients = list()
            interaction_trust_vals = list()
            for j in range(total_clients):
                if current_system_trust_mat[i][j] == 0.01:
                    zero_interaction_clients.append(j)
                else:
                    interaction_trust_vals.append(current_system_trust_mat[i][j])
            
            if len(zero_interaction_clients) == total_clients:
                not_selected_before_starting += 1
            interaction_trust_mean = np.mean(np.array(interaction_trust_vals))
            interaction_trust_std = np.std(np.array(interaction_trust_vals))
            sampled_interaction_trust_vals = list(np.random.normal(interaction_trust_mean, interaction_trust_std, len(zero_interaction_clients)))
            for j, sampled_trust in zip(zero_interaction_clients, sampled_interaction_trust_vals):
                current_system_trust_mat[i][j] = sampled_trust
            ## client i would have 100% trust on itself
            current_system_trust_mat[i][i] = (1+1)/200

    print(f"{not_selected_before_starting} clients were not selected in starting rounds")
    ## also we give 100% trust between intial_validation_clients, since we starting with same initial trust 0.01
    ## between all clients, we just add this extra trust
    num_validating_clients = len(initial_validation_clients)
    for i in range(num_validating_clients):
        for j in range(i+1, num_validating_clients):
            current_system_trust_mat[initial_validation_clients[i]][initial_validation_clients[j]] += (1+1)/200
            current_system_trust_mat[initial_validation_clients[j]][initial_validation_clients[i]] += (1+1)/200
    
    ## renormalizing the trust matrix
    sum_of_rows = current_system_trust_mat.sum(axis=1)
    current_system_trust_mat = current_system_trust_mat / sum_of_rows[:, np.newaxis]



def fill_initial_trust(computing_clients):
    global config
    global initial_aggregate_grads
    global current_system_trust_mat
    global current_system_trust_vec

    ## added values in the trust matrix selected in the round
    num_computing_clients = len(computing_clients)
    for i in range(num_computing_clients):
            comp1_vec = copy.deepcopy(initial_aggregate_grads[computing_clients[i]]).reshape(1, -1)
            comp1_vec /= np.linalg.norm(comp1_vec)
            for j in range(i+1, num_computing_clients):
                comp2_vec = copy.deepcopy(initial_aggregate_grads[computing_clients[j]]).reshape(1, -1)
                comp2_vec /= np.linalg.norm(comp2_vec)
                new_trust_val = (1+cosine_similarity(comp1_vec, comp2_vec)[0][0])/200

                prev_val_ij = current_system_trust_mat[computing_clients[i], computing_clients[j]]
                current_system_trust_mat[computing_clients[i], computing_clients[j]] = prev_val_ij*config['BETA'] + (1-config['BETA'])*new_trust_val

                prev_val_ji = current_system_trust_mat[computing_clients[j], computing_clients[i]]
                current_system_trust_mat[computing_clients[j], computing_clients[i]] = prev_val_ji*config['BETA'] + (1-config['BETA'])*new_trust_val
    

    ## we are also updating our trust vec using joint grad vec from that round
    agg_val_vector = torch.zeros(initial_aggregate_grads[computing_clients[0]].size())
    ## join grad vector from all validation client and use that for cosine similarity for all computing clients
    for comp_client in computing_clients:
        agg_val_vector += copy.deepcopy(initial_aggregate_grads[comp_client])
    
    agg_val_vector = agg_val_vector.reshape(1, -1)
    agg_val_vector /= np.linalg.norm(agg_val_vector)
    
    ## now we iterate over the computing clients to give them trust values\
    for comp_client in computing_clients:
        comp_vec = copy.deepcopy(initial_aggregate_grads[comp_client]).reshape(1, -1)
        comp_vec /= np.linalg.norm(comp_vec)
        new_trust_val = (1+cosine_similarity(comp_vec, agg_val_vector)[0][0])/200
        prev_val = current_system_trust_vec[comp_client]
        current_system_trust_vec[comp_client] = config['ALPHA']*prev_val + (1-config['ALPHA'])*new_trust_val

    ## we don't normalize trust matrix and trust vec here, we do it only once before starting cos_defence
    

def cos_defence(computing_clients, poisoned_clients, threshold=0.0):
    ## here poisoned clients are just used for analytics purpose
    global config
    
    ## for trust plots we collect this data
    global all_trust_vals
    global all_client_types
    global client_ids
    global validation_client_ids

    global current_system_trust_mat
    global current_system_trust_vec

    if config['COLLAB_ALL']:
        validating_clients = computing_clients
    else:
        ## first identify validation clients from computing clients, we assume half of the computing clients as
        ## validation clients, top k values
        system_trust_vals = np.array(current_system_trust_vec.copy())
        selected_client_trust_vals = system_trust_vals[computing_clients]
        val_client_num = int(len(computing_clients)/2)
        validating_clients = computing_clients[np.argsort(selected_client_trust_vals)[-val_client_num:]]
    
    logging.info(f"Validating client selected: {validating_clients}")
    

    ## update trust matrix
    # Step 4, 5 Computing trust and updating the system trust matrix,
    if config['COLLAB_MODE']:
        global aggregate_grads

        agg_val_vector = torch.zeros(aggregate_grads[validating_clients[0]].size())
        ## join grad vector from all validation client and use that for cosine similarity for all computing clients
        for val_client in validating_clients:
            agg_val_vector += copy.deepcopy(aggregate_grads[val_client])
        
        agg_val_vector = agg_val_vector.reshape(1, -1)
        agg_val_vector /= np.linalg.norm(agg_val_vector)
        
        ## now we iterate over the computing clients to give them trust values
        comp_trusts = list()
        for comp_client in computing_clients:
            comp_vec = copy.deepcopy(aggregate_grads[comp_client]).reshape(1, -1)
            comp_vec /= np.linalg.norm(comp_vec)
            comp_trusts.append((1+cosine_similarity(comp_vec, agg_val_vector)[0][0])/2)

        trust_arr = np.array(comp_trusts).reshape(-1, 1)

        ##This implementation needs to be done for individual mode scenario.
        if config['TRUST_CUT_METHOD'] == 0:
            ## make 2 clusters assuming one for honest (with more number of clients) and one for malicious (with low number)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(trust_arr)
            kmeans_centers = kmeans.cluster_centers_
            center_dist = abs(kmeans_centers[0] - kmeans_centers[1])
            ## implement the logic here.
            # logging.info(f"Trust cluster diff: {center_dist}")
            # honest_trust_threshold = np.max(kmeans_centers)*(1 - config['HONEST_PARDON_FACTOR']*center_dist)
        elif config['TRUST_CUT_METHOD'] == 1:
            ## AFA method
            mean_val = np.mean(trust_arr)
            median_val = np.median(trust_arr)
            std_dev = np.std(trust_arr)
            if mean_val >= median_val:
                honest_trust_threshold = median_val + config['HONEST_PARDON_FACTOR']*std_dev
                trust_arr = np.where(trust_arr <= honest_trust_threshold, trust_arr, 0.002)
            else:
                honest_trust_threshold = median_val - config['HONEST_PARDON_FACTOR']*std_dev
                trust_arr = np.where(trust_arr >= honest_trust_threshold, trust_arr, 0.002)
        else:
            honest_trust_threshold = 0.0
            trust_arr = np.where(trust_arr >= honest_trust_threshold, trust_arr, 0.002)
        
        comp_trusts = list(trust_arr)
        for val_client, comp_client, new_trust_val in zip(validating_clients, computing_clients, comp_trusts):
            prev_val = current_system_trust_mat[val_client, comp_client]
            current_system_trust_mat[val_client, comp_client] = prev_val*config['BETA'] + (1-config['BETA'])*new_trust_val
            
            ## for analytics purposes
            all_trust_vals.append(new_trust_val)
            client_ids.append(comp_client)
            if comp_client in poisoned_clients:
                client_type = "minor_offender"
                if comp_client//20 == 1:
                    client_type = "major_offender"
            else:
                client_type = "honest"
            all_client_types.append(client_type)
    else:
        for val_client in validating_clients:
            val_vec = copy.deepcopy(aggregate_grads[val_client]).reshape(1, -1)
            val_vec /= np.linalg.norm(val_vec)
            for comp_client in computing_clients:
                if comp_client != val_client:
                    comp_vec = copy.deepcopy(aggregate_grads[comp_client]).reshape(1, -1)
                    comp_vec /= np.linalg.norm(comp_vec)
                    new_trust_val = (1+cosine_similarity(comp_vec, val_vec)[0][0])/200

                    prev_val = current_system_trust_mat[val_client, comp_client]
                    current_system_trust_mat[val_client, comp_client] = prev_val*config['BETA'] + (1-config['BETA'])*new_trust_val
                    
                    ## for analytics purposes
                    all_trust_vals.append(new_trust_val)
                    client_ids.append(comp_client)
                    validation_client_ids.append(val_client)
                    if comp_client in poisoned_clients:
                        client_type = "minor_offender"
                        if comp_client//20 == 1:
                            client_type = "major_offender"
                    else:
                        client_type = "honest"
                    all_client_types.append(client_type) 
            
    ## we need to normalize the new_system_trust_mat row wise
    sum_of_rows = current_system_trust_mat.sum(axis=1)
    current_system_trust_mat = current_system_trust_mat / sum_of_rows[:, np.newaxis]
    # new_system_trust_mat = new_system_trust_mat / new_system_trust_mat.max(axis=0)

    # Step 6
    # Now we calculate new system trust vector with the help of eigen_trust, since the variables are global
    # we update directly to the current_system_trust_vec
    eigen_trust(config['ALPHA'])

    # now aggregate the new global model using this system trust vec
    # in normal_scenario all clients get equal weight but we have given trust vector based on
    # the computation.
    # Thresholding: If we want to set a threhold for minimum trust in order to taken into model averaging
    ## i.e. setting all trust values below threshold = 0.01 to zero.
    current_system_trust_vec = np.where(current_system_trust_vec > threshold, current_system_trust_vec, 0.0)



def train_on_client(idx, model, data_loader, optimizer, loss_fn, device):
    global config
    global sel_layer_names
    model.train()
    epoch_training_losses = []

    epoch_grad_bank = dict()
    for layer_name in sel_layer_names:
        epoch_grad_bank[f'{layer_name}.weight'] = torch.zeros(getattr(model, layer_name).weight.size())
        epoch_grad_bank[f'{layer_name}.bias'] = torch.zeros(getattr(model, layer_name).bias.size())


    for epoch in range(config['LOCAL_EPOCHS']):
        train_loss = 0.0
        for data, target in data_loader:
            # move tensors to GPU device if CUDA is available
            data, target = data.to(device), target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = loss_fn(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            for layer_name in sel_layer_names:
                epoch_grad_bank[f'{layer_name}.weight'] += getattr(model, layer_name).weight.grad.detach().clone().cpu()
                epoch_grad_bank[f'{layer_name}.bias'] += getattr(model, layer_name).bias.grad.detach().clone().cpu()


            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()

        epoch_train_loss = train_loss / len(data_loader)
        epoch_training_losses.append(epoch_train_loss)
        logging.debug('Client: {}\t Epoch: {} \tTraining Loss: {:.6f}'.format(idx, epoch, epoch_train_loss))

    
    
    ## for first 10 iterations we save all gradients layerwise to find important feature using gradients
    global save_for_feature_finding
    global grad_bank
    global collect_features
    global initial_aggregate_grads
    global aggregate_grads
    global indicative_grads

    ## we save initial grad vector of clients to find similarity, to initialize trust matrix
    if collect_features:
        ## save here what you want to access later for similarity calculation, for e.g. last layer params of the model
        ## or calculated by clustering method to detect important features
        epoch_grad_vecs = []
        for key in epoch_grad_bank.keys():
            layer_grads = (epoch_grad_bank[key]/config['LOCAL_EPOCHS']).numpy()
            # logging.info("Printing shapes")
            # logging.info(layer_grads.size())
            # logging.info(indicative_grads[key].shape)
            epoch_grad_vecs.append(layer_grads[indicative_grads[key].astype(bool)].flatten().reshape(1, -1))
        
        # we just add new params to old ones, during similarity calculation we anyway normalize the whole vector
        # grad_vec = 
        aggregate_grads[idx] += np.concatenate(epoch_grad_vecs, axis=1)
    else:
        ## we save initial grad vector of clients to find similarity, to initialize trust matrix
        epoch_grad_vecs = []
        for key in epoch_grad_bank.keys():
            layer_grads = (epoch_grad_bank[key]/config['LOCAL_EPOCHS']).detach().clone().cpu().numpy()
            # logging.info("Printing shapes")
            # logging.info(layer_grads.size)
            epoch_grad_vecs.append(layer_grads.reshape(1, -1))

            ## Needed for auror, auror_plus algo
            if save_for_feature_finding:
                grad_bank[key].append(epoch_grad_bank[key]/ config['LOCAL_EPOCHS'])
        
        # we just add new params to old ones, during similarity calculation we anyway normalize the whole vector 
        initial_aggregate_grads[idx] += np.concatenate(epoch_grad_vecs, axis=1)

    return epoch_training_losses


def run_test(model, test_data_loader, loss_fn, device):
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    # specify the image classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    model.eval()

    # Need to save these in case of final test
    predictions = []
    ground_truths = []
    attack_scount = 0
    attack_srate = 0
    source_class = 2
    target_class = 9
    # iterate over test data
    for data, target in test_data_loader:
        data, target = data.to(device), target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = loss_fn(output, target)
        # update test loss
        test_loss += loss.item()
        # convert output probabilities to predicted class
        top_p, pred_class = torch.max(output, 1)
        predictions.extend(pred_class.tolist())
        ground_truths.extend(target.tolist())
        # compare predictions to true label
        correct_tensor = pred_class.eq(target.data.view_as(pred_class))
        correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(
            correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
            ## checking if attack was successful
            if ground_truths[i] == source_class and pred_class[i] == target_class:
                attack_scount += 1
    
    ## just making sure we had some example of source class
    if class_total[source_class] > 0:
        attack_srate =(100 * attack_scount)/class_total[source_class]
    logging.info(f"Attack Success Rate: {attack_srate}")
    # average test loss
    test_loss = test_loss / len(test_data_loader.dataset)
    logging.debug('Test Loss: {:.6f}\n'.format(test_loss))

    all_classes_acc = []
    for i in range(10):
        if class_total[i] > 0:
            # cls_acc = 'Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            #     classes[i], (100 * class_correct[i]) / class_total[i],
            #     np.sum(class_correct[i]), np.sum(class_total[i]))
            cls_acc = (100 * class_correct[i]) / class_total[i]
        else:
            # cls_acc = 'Test Accuracy of %5s: N/A (no training examples)' % (classes[i])
            cls_acc = -1
        logging.info(cls_acc)
        all_classes_acc.append(cls_acc)
    total_acc = 100. * np.sum(class_correct) / np.sum(class_total)
    logging.info('\nTest Accuracy (Overall): %2d%% (%2d/%2d)\n\n' % (
        total_acc, np.sum(class_correct), np.sum(class_total)))

    return test_loss, total_acc, all_classes_acc, predictions, ground_truths, attack_srate


def fed_avg(server_model, selected_clients, client_models, client_weights):
    # Safety lock, to not update model params accidentally
    with torch.no_grad():
        # need to take avg key-wise
        for key in server_model.state_dict().keys():
            temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
            for sel_client in selected_clients:
                temp += client_weights[sel_client] * client_models[sel_client].state_dict()[key]
            # update the new value of this key in the server model
            server_model.state_dict()[key].data.copy_(temp)
            # update this key value in all the client models as well
            for idx in range(len(client_weights)):
                client_models[idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, client_models


def print_trust_vals(trust_vals):
    trust1, trust2, trust12 = 0, 0, 0
    count = 0
    for item in trust_vals:
        trust1 += item[0][0]
        trust2 += item[1][0]
        trust12 += item[2][0]
        count += 1
    logging.info(f"trust1: {trust1/count}, trust2: {trust2/count} and trust12: {trust12/count}")


def gen_trust_plots(client_ids, validation_client_ids, trust_vals, labels):
    global config
    global base_path
    save_location = os.path.join(base_path, 'results/plots/')
    dataframe_location = os.path.join(base_path, 'results/plot_dfs/')
    current_time = time.localtime()
    config_details = f"{config['DATASET']}_C{config['CLIENT_FRAC']}_P{config['POISON_FRAC']}_FDRS{config['FED_ROUNDS']}_CDF{config['COS_DEFENCE']}_SEL{config['SEL_METHOD']}_CLB{config['COLLAB_MODE']}_LYRS{config['CONSIDER_LAYERS']}_FFA{config['FEATURE_FINDING_ALGO']}_CSEP{config['CLUSTER_SEP']}"

    if config['COLLAB_MODE']:
        ## since in COLLAB_MODE multiple validation clients give trust value we don't have 1:1 ref for
        ## computing client who gave them trust value
        trust_data ={'client_id': client_ids, 'trust_val': trust_vals, 'client_label': labels}
    else:
        trust_data ={'client_id': client_ids, 'validation_client_id': validation_client_ids, 'trust_val': trust_vals, 'client_label': labels}
    
    trust_df = pd.DataFrame.from_dict(trust_data)
    trust_df['modified_trust'] = trust_df['trust_val'].apply(lambda x: int(x*10000))
    trust_df.to_pickle(f'{dataframe_location}trust_{config_details}_{time.strftime("%Y-%m-%d %H:%M:%S", current_time)}.pkl')


    ## 1 D Data strip
    strip_fig = px.strip(trust_df, x="modified_trust", y="client_label", color="client_label", color_discrete_map={
                "honest": '#00CC96',
                "minor_offender": '#EF553B',
                "major_offender": '#AB63FA'})
    strip_fig.update_layout(title='Trust given by validation clients')
    strip_fig.write_html(os.path.join(save_location,'{}_trust_strip_{}.html'.format(config_details, time.strftime("%Y-%m-%d %H:%M:%S", current_time))))

    ## histogram of trust vals
    histo_fig = px.histogram(trust_df, x="modified_trust", color="client_label", color_discrete_map={
                "honest": '#00CC96',
                "minor_offender": '#EF553B',
                "major_offender": '#AB63FA'})
    histo_fig.update_layout(title='Trust given by validation clients', barmode="group")
    histo_fig.write_html(os.path.join(save_location,'{}_trust_histo_{}.html'.format(config_details, time.strftime("%Y-%m-%d %H:%M:%S", current_time))))

def gen_trust_curves(trust_scores, initial_validation_clients, poisoned_clients, start_cosdefence):
    ## this function generate curves to show how trust of different clients change with every fedrounds
    ## only showing iterations when cos_defence starts changing trusts
    ## horizontal lines means either they were not selected in those rounds or the trust variation is very low
    global config
    trust_scores = (np.array(trust_scores)*10000).astype(int)
    client_types = ["honest" for _ in range(config['TOTAL_CLIENTS'])]
    for val_client in initial_validation_clients:
        client_types[val_client] = "init_val"
    for p_client in poisoned_clients:
        client_types[p_client] = "poisoned"

    # client_types = np.zeros(config['TOTAL_CLIENTS'], dtype=int)
    # client_types[initial_validation_clients] = -1
    # client_types[poisoned_clients] = 1
    logging.debug(f"client_types: {client_types}")

    fdrs = config['FED_ROUNDS']
    trust_scores = trust_scores[start_cosdefence:]

    trust_scores_df = pd.DataFrame(columns=["fed_round","trust_score", "client_id", "client_type"])
    for client in range(config['TOTAL_CLIENTS']):
        temp_df = pd.DataFrame({"fed_round" : pd.Series(list(range(start_cosdefence, fdrs)), dtype="int"),
                                "trust_score" : trust_scores[:, client],
                                "client_id" : client,
                                "client_type":client_types[client]})
        trust_scores_df = trust_scores_df.append(temp_df, ignore_index=True)
    logging.debug(trust_scores_df.shape)

    ## plotting
    global base_path
    dataframe_location = os.path.join(base_path, 'results/plot_dfs/')
    current_time = time.localtime()
    config_details = f"{config['DATASET']}_C{config['CLIENT_FRAC']}_P{config['POISON_FRAC']}_FDRS{config['FED_ROUNDS']}_CDF{config['COS_DEFENCE']}_SEL{config['SEL_METHOD']}_CLB{config['COLLAB_MODE']}_LYRS{config['CONSIDER_LAYERS']}_FFA{config['FEATURE_FINDING_ALGO']}_CSEP{config['CLUSTER_SEP']}"
    trust_scores_df.to_pickle(f'{dataframe_location}trust_scores_{config_details}_{time.strftime("%Y-%m-%d %H:%M:%S", current_time)}.pkl')
    
    save_location = os.path.join(base_path, 'results/plots/')
    score_curves_fig = px.line(trust_scores_df, x="fed_round", y="trust_score", color="client_type",
                                line_group="client_id", hover_name="client_id", color_discrete_map={
                                "init_val": '#00CC96',
                                "honest": '#636EFA',
                                "poisoned": '#EF553B'})
    score_curves_fig.update_layout(title="Trust Score Evolution")
    score_curves_fig.write_html(os.path.join(save_location,'{}_trust_score_curves_{}.html'.format(config_details, time.strftime("%Y-%m-%d %H:%M:%S", current_time))))

def gen_accuracy_poison_data_plot(attack_srates, source_class_accuracy, total_accuracy, all_poisoned_client_selected):
    global config
    if config['DATASET'] == "mnist":
        if config['CLASS_RATIO'] == 10:
            minor_class_examples = 28
        elif config['CLASS_RATIO'] == 4:
            minor_class_examples = 41
        else:
            minor_class_examples = 54
    elif config['DATASET'] == "fmnist":
        # in fashion mnist all classes have 6000 examples each
        if config['CLASS_RATIO'] == 10:
            minor_class_examples = 31
        elif config['CLASS_RATIO'] == 4:
            minor_class_examples = 46
        else:
            minor_class_examples = 60
    else:
        if config['CLASS_RATIO'] == 10:
            minor_class_examples = 26
        elif config['CLASS_RATIO'] == 4:
            minor_class_examples = 38
        else:
            minor_class_examples = 50
    major_class_examples = minor_class_examples * config['CLASS_RATIO']
    ## need to define this for other datasets as well

    poisoned_examples = list()
    for fed_round in range(config['TEST_EVERY']-1, config['FED_ROUNDS'], config['TEST_EVERY']):
        poisoned_clients_in_round = all_poisoned_client_selected[fed_round]
        poisoned_data_in_round = 0
        for pclient in poisoned_clients_in_round:
            if pclient//20 == 1:
                poisoned_data_in_round += major_class_examples
            else:
                poisoned_data_in_round += minor_class_examples
        poisoned_examples.append(poisoned_data_in_round)

    ## saving this plot data in order to later compare two plots if we need to
    global base_path
    current_time = time.localtime()
    config_details = f"{config['DATASET']}_C{config['CLIENT_FRAC']}_P{config['POISON_FRAC']}_FDRS{config['FED_ROUNDS']}_CDF{config['COS_DEFENCE']}_SEL{config['SEL_METHOD']}_CLB{config['COLLAB_MODE']}_LYRS{config['CONSIDER_LAYERS']}_FFA{config['FEATURE_FINDING_ALGO']}_CSEP{config['CLUSTER_SEP']}"
    
    testing_round = list(range(config['TEST_EVERY']-1, config['FED_ROUNDS'], config['TEST_EVERY']))
    dataframe_location = os.path.join(base_path, 'results/plot_dfs/')
    accuracy_poison_df = pd.DataFrame()
    accuracy_poison_df['testing_round'] = testing_round
    accuracy_poison_df['total_accuracy'] = total_accuracy
    accuracy_poison_df['source_class_accuracy'] = source_class_accuracy
    accuracy_poison_df['attack_srates'] = attack_srates
    accuracy_poison_df.to_pickle(f'{dataframe_location}accuracy_poison_{config_details}_{time.strftime("%Y-%m-%d %H:%M:%S", current_time)}.pkl')
    
    ## plotting
    save_location = os.path.join(base_path, 'results/plots/')
    acc_poison_fig = make_subplots(rows=3, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.01)

    acc_poison_fig.add_trace(go.Scatter(name='Total Accuracy', x=testing_round, y=total_accuracy, mode='lines+markers'),
                                row=1, col=1)
    acc_poison_fig.add_trace(go.Scatter(name='Poisoned Class Accuracy', x=testing_round, y=source_class_accuracy, mode='lines+markers'),
                                row=1, col=1)

    acc_poison_fig.add_trace(go.Bar(name='Poisoned Examples', x=testing_round, y=poisoned_examples),
                                row=2, col=1)

    acc_poison_fig.add_trace(go.Scatter(name='Attack Success Rate', x=testing_round, y=attack_srates, mode='lines+markers'),
                                row=3, col=1)

    acc_poison_fig.update_layout(height=800, width=1200, colorway=['#636EFA', '#EF553B', '#DC587D', '#00B5F7'],
                                title_text="Accuracy variations with poisoned examples")
    
    acc_poison_fig.write_html(os.path.join(save_location,'{}_acc_poison_plot_{}.html'.format(config_details, time.strftime("%Y-%m-%d %H:%M:%S", current_time))))





def trust_clustering(trust_vals, labels):
    trust_arr = np.array(trust_vals).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(trust_arr)
    logging.info(kmeans.cluster_centers_)
    

def start_fl(with_config):
    global config
    config = with_config
    ## config short summary
    config_ss = f"{config['DATASET']}_C{config['CLIENT_FRAC']}_P{config['POISON_FRAC']}_FDRS{config['FED_ROUNDS']}_CDF{config['COS_DEFENCE']}_{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    
    global base_path
    logs_folder = os.path.join(base_path, 'logs/')
    Path(logs_folder).mkdir(parents=True, exist_ok=True)
    logs_file = logs_folder + config_ss +'.log'
    logging.basicConfig(filename=logs_file, level=getattr(logging, config['LOG_LEVEL']))
    logging.info(config)



    ### initializing global variables block start ###
    global current_system_trust_vec
    global current_system_trust_mat

    ##global variables !! check when running multiple experiments together
    current_system_trust_vec = set_initial_trust_vec("ones")

    ## inital system trust need to be set using three type of distributions
    current_system_trust_mat = set_initial_trust_mat("ones")  #

    global initial_aggregate_grads
    global aggregate_grads
    global grad_bank
    global indicative_grads
    global save_for_feature_finding
    global collect_features

    aggregate_grads = list()
    grad_bank = dict()
    indicative_grads = dict()
    save_for_feature_finding = False
    collect_features = False

    global client_ids
    global validation_client_ids 
    global all_trust_vals
    global all_client_types
    
    client_ids = list()
    validation_client_ids = list() 
    all_trust_vals = list()
    all_client_types = list()

    ### End of global variable initializing block

    ## trust scores of the system after every fed rounds are saved in this list
    trust_scores = list()

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Computing Device:{device}")
    if config['RANDOM_PROCESS']:
        rng1 = default_rng()
        rng2 = default_rng()
    else:
        seed = 42
        rng1 = default_rng(seed)
        rng2 = default_rng(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    # If this flag is set first client data is created
    if config['CREATE_DATASET']:
        create_client_data(config['RANDOM_DATA'], config['DATASET'], config['CLASS_RATIO'])

    # if config['COS_DEFENCE']:
    if config['GRAD_COLLECT_FOR'] == -1:
        ## -1 here tells that cos_defence should be started based on the dataset
        if config['DATASET'] == 'mnist':
            start_cosdefence = config['GRAD_COLLECTION_START'] + int(1/config['CLIENT_FRAC'])
        elif config['DATASET'] == 'fmnist':
            start_cosdefence = config['GRAD_COLLECTION_START'] + 2 * int(1/config['CLIENT_FRAC'])
        else:
            start_cosdefence = config['GRAD_COLLECTION_START'] + 4 * int(1/config['CLIENT_FRAC'])
    else:
        start_cosdefence = config['GRAD_COLLECTION_START'] + config['GRAD_COLLECT_FOR']


    ## this will return model based on selection
    server_model = get_model(config['MODEL'])
    

    # using gpu for computations if available
    server_model = server_model.to(device)

    ## initializing global grad bank, based on model and layers selected
    global sel_layer_names
    sel_layer_names = get_selected_layers(server_model.layer_names, config['CONSIDER_LAYERS'])
    for layer_name in sel_layer_names:
        grad_bank[layer_name + '.weight'] = list()
        grad_bank[layer_name + '.bias'] = list()
    
    initial_aggregate_grads = list()
    init_grad_vec_size = 0
    ## need to make based on the layer selected
    for layer_name in sel_layer_names:
        init_grad_vec_size += torch.numel(getattr(server_model, layer_name).weight)
        init_grad_vec_size += torch.numel(getattr(server_model, layer_name).bias)
    # print(f"Initial grad vec size: { init_grad_vec_size}")
    for client in range(config['TOTAL_CLIENTS']):
        initial_aggregate_grads.append(torch.zeros((1, init_grad_vec_size)))



    # specify loss function (categorical cross-entropy)
    loss_fn = nn.CrossEntropyLoss()

    # choose how many clients you want send model to
    total_clients = 100
    client_models = [copy.deepcopy(server_model).to(device) for _idx in range(config['TOTAL_CLIENTS'])]

    # location of data with the given config
    data_folder = os.path.join(base_path, f"data/{config['DATASET']}/fed_data/label_flip0/poisoned_{int(config['POISON_FRAC']*100)}CLs/")

    # specify learning rate to be used
    if config['COS_DEFENCE']:
        ## we'll increase it, after cos_defence starts working
        learning_rate = config['LEARNING_RATE']/config['REDUCE_LR']
    else:
        learning_rate = config['LEARNING_RATE']  # change this according to our model, tranfer learning use 0.001, basic model use 0.01
    if config['OPTIMIZER'] == 'sgd':
        optimizers = [optim.SGD(params=client_models[idx].parameters(), lr=config['LEARNING_RATE']) for idx in range(config['TOTAL_CLIENTS'])]
    else:
        optimizers = [optim.Adam(params=client_models[idx].parameters(), lr=config['LEARNING_RATE']) for idx in range(config['TOTAL_CLIENTS'])]
    
    client_data_loaders = create_client_data_loaders(config['TOTAL_CLIENTS'], data_folder, config['BATCH_SIZE'])
    test_data_loader = get_test_data_loader(config['DATASET'], config['BATCH_SIZE'])

    # Poisoned clients in this setting
    poisoned_clients = []
    poison_config_file = data_folder + 'poison_config.txt'
    with open(poison_config_file, 'r') as pconfig_file:
        pinfo_data = json.load(pconfig_file)
        poisoned_clients = pinfo_data['poisoned_clients']
        ## printing poisoned client composition in the environment
        major_offender_count = 0
        for poisoned_client in poisoned_clients:
            if poisoned_client//20 == 1:
                major_offender_count += 1
        print(f"Major offender: {major_offender_count}, Minor offender: {len(poisoned_clients)-major_offender_count}")

    if config['COS_DEFENCE']:
        initial_validation_clients = init_validation_clients(poisoned_clients, rng2)

    ###Training and testing model every kth round
    total_accs = []
    source_class_accs = []
    attack_srates = []
    classes_accs = []
    classes_f1scores = []
    avg_metric_vals = []
    poisoned_clients_sel_in_rounds = []
    client_training_losses = [[] for i in range(config['TOTAL_CLIENTS'])]
    avg_training_losses = [] # this saves the avg loss of the clients selected in one federated round

    ## boolean flag used to check if trust mat and vec was initialized using similarity between grads of clients
    trust_initialized = False
    selected_hq = np.ones((config['TOTAL_CLIENTS']), dtype=float)

    ###
    ### Actual federated learning starts here
    ###
    for i in range(config['FED_ROUNDS']):

        ## selecting clients based on probability or always choose clients with highest trust
        logging.debug(f"System trust vec sum: {current_system_trust_vec.sum()}")
        trust_scores.append(current_system_trust_vec.copy())
        if config['COS_DEFENCE']:
            ## increase the learning rate now
            if i == start_cosdefence:
                for optimizer in optimizers:
                    for op_grp in optimizer.param_groups:
                        op_grp['lr'] = config['LEARNING_RATE']

            if i >= start_cosdefence:       
                if config['SEL_METHOD'] == 0:
                    clients_selected = rng1.choice(config['TOTAL_CLIENTS'], size=int(config['TOTAL_CLIENTS'] * config['CLIENT_FRAC']), replace=False)
                elif config['SEL_METHOD'] == 1:
                    clients_selected = rng1.choice(config['TOTAL_CLIENTS'], p=current_system_trust_vec, size=int(config['TOTAL_CLIENTS'] * config['CLIENT_FRAC']), replace=False)
                else:
                    top_trust_indices = np.argsort(current_system_trust_vec + selected_hq)[-(int(config['TOTAL_CLIENTS'] * config['CLIENT_FRAC'])):]
                    ## since out client ids are also numbered from 0 to 99
                    clients_selected = top_trust_indices
                    ## increase the history quotient for all clients by a factor of 0.005, this factor
                    ## that if a client wants to be get selected each time he has to do better than 50% 
                    ## similarity
                    selected_hq += 0.009
                    ## now for the selected clients make this value 1.0 (initial val)
                    np.put(selected_hq, clients_selected, 1.0)
            else:
                ## until cos_defence starts we want as much different clients to be selected to effectively fill trust
                ## matrix and trust vec
                clients_selected = rng1.choice(config['TOTAL_CLIENTS'], size=int(config['TOTAL_CLIENTS'] * config['CLIENT_FRAC']), replace=False)
        else:
            ## when cos_defence is off then we always select randomly, to avoid selecting same clients every time
            clients_selected = rng1.choice(config['TOTAL_CLIENTS'], size=int(config['TOTAL_CLIENTS'] * config['CLIENT_FRAC']), replace=False)


        logging.info(f"selected clients in round {i}: {clients_selected}")

        poisoned_clients_selected = list(set(poisoned_clients) & set(clients_selected))
        
        logging.info(f"poisoned clients in round {i}: {poisoned_clients_selected}")
        poisoned_clients_sel_in_rounds.append(poisoned_clients_selected)

        temp_training_losses = []
        for j in clients_selected:
            training_losses = train_on_client(j, client_models[j], client_data_loaders[j], optimizers[j], loss_fn, device)
            client_training_losses[j].extend(training_losses)
            temp_training_losses.append(sum(training_losses)/config['LOCAL_EPOCHS'])

        avg_training_losses.append(sum(temp_training_losses)/len(clients_selected))

        # if turned on we change the client_weights from normal to computed by CosDefence
        logging.info(f"CosDefence is On: {config['COS_DEFENCE']}")
        if config['COS_DEFENCE']:
            if i == config['GRAD_COLLECTION_START']:
                save_for_feature_finding = True
            elif i == start_cosdefence - 1:
                indicative_grads, counts = find_indicative_grads(grad_bank, config['FEATURE_FINDING_ALGO'], config['CLUSTER_SEP'])
                save_for_feature_finding = False
                collect_features = True

                ## to make sure when client are selected this value sums upto 1
                current_system_trust_vec /= current_system_trust_vec.sum()
                
                ## this code is upload pre-calculated grad features.
                # layer_names = ['fc1', 'fc2', 'output_layer']
                # counts = 0
                # for name in layer_names:
                #     bias_arr = np.load(name + '.bias.npy')
                #     weight_arr = np.load(name + '.weight.npy')
                #     logging.info(f"Indicative grad of {name} has sizes")
                #     logging.info(bias_arr.shape)
                #     logging.info(weight_arr.shape)
                #     indicative_grads[name + '.bias'] = bias_arr
                #     indicative_grads[name + '.weight'] = weight_arr
                #     counts += np.count_nonzero(bias_arr)
                #     counts += np.count_nonzero(weight_arr)
                
                ## initializing aggregate grads so that now these grads can ve collected as flat vector
                for k in range(config['TOTAL_CLIENTS']):
                    aggregate_grads.append(torch.zeros((1, counts)))

                logging.info(f"Found {counts} indicative grads")

            elif i >= start_cosdefence:
                ## before starting cos_defence, fill up matrix and vec using mean, std method
                if not trust_initialized:
                    fill_up_rem_trust_mat_and_vec(config['TOTAL_CLIENTS'], initial_validation_clients, fill_up=True)
                    trust_initialized = True
                cos_defence(clients_selected, poisoned_clients_selected, 0)
            else:
                ## calculate similarity between clients from initial_aggregate_grads, to fill up trust matrix and trust vec
                fill_initial_trust(clients_selected)


        ## New weight setting strategy, if cos_defence is on then it modifies current_system_trust_vec, meaning
        ## it changes the weights of the client selected, if not initial trust vec will be used.
        if config['COS_DEFENCE'] and i >= start_cosdefence:
            client_weights = np.copy(current_system_trust_vec)
            client_weights = torch.from_numpy(client_weights)
            
            ## due to different type of initialization client weights remain low
            ## to correct this we renormalize the weights of the selected clients, so that their sum would be 1
            weights = np.zeros(len(clients_selected), dtype=float)
            for j, client in enumerate(clients_selected):
                weights[j] = client_weights[client]
            # logging.info(f"Complete Trust vec {current_system_trust_vec}")
            logging.info(f"Trust value on computing cleints{current_system_trust_vec[clients_selected]}")
            logging.info(weights)
            weights = weights/weights.sum()

            for j, client in enumerate(clients_selected): 
                client_weights[client] = weights[j]
            client_weights = client_weights.to(device)
        else:
            ## Earlier weight setting strategy
            client_weights = [1 / (len(clients_selected)) for i in range(config['TOTAL_CLIENTS'])]  # need to check about this
            client_weights = torch.tensor(client_weights)

        # aggregate to update server_model and client_models
        server_model, client_models = fed_avg(server_model, clients_selected, client_models, client_weights)
        logging.info(f"Round {i} complete")
    
        # Testing Model every kth round
        if (i + 1) % config['TEST_EVERY'] == 0:
            testing_loss, total_acc, classes_acc, predictions, ground_truths, attack_srate = run_test(server_model, test_data_loader, loss_fn, device)
            cls_precisions, cls_recalls, cls_f1scores, cls_supports = metrics.precision_recall_fscore_support(ground_truths, predictions, average=None, zero_division=1)
            # classes_precisions.append(cls_precisions.tolist())
            # classes_recalls.append(cls_recalls.tolist())
            classes_f1scores.append(cls_f1scores.tolist())
            # classes_supports.append(cls_supports.tolist())
            # we store "weighted" average values of precision, recall, f1score and support in this list.
            avg_metric_vals.append(metrics.precision_recall_fscore_support(ground_truths, predictions, average='weighted', zero_division=1))
            # testing_losses.append(testing_loss)
            ## saving total and source classes accuracy separately which we can plot later.
            total_accs.append(total_acc)
            source_class_accs.append(classes_acc[2])
            attack_srates.append(attack_srate)
            classes_accs.append(classes_acc)


    ## generating various plots
    plots_folder = os.path.join(base_path, 'results/plots/')
    Path(plots_folder).mkdir(parents=True, exist_ok=True)
    dataframe_location = os.path.join(base_path, 'results/plot_dfs/')
    Path(dataframe_location).mkdir(parents=True, exist_ok=True)
    gen_accuracy_poison_data_plot(attack_srates, source_class_accs, total_accs, poisoned_clients_sel_in_rounds)
    if config['COS_DEFENCE']:
        gen_trust_plots(client_ids, validation_client_ids, all_trust_vals, all_client_types)
        gen_trust_curves(trust_scores, initial_validation_clients, poisoned_clients, start_cosdefence)
        trust_clustering(all_trust_vals, all_client_types)

    mean_poison_class_acc = np.mean(np.array(source_class_accs[start_cosdefence:]))
    mean_attack_srate = np.mean(np.array(attack_srates[start_cosdefence:]))
    logging.info(f"Mean Poison class accuracy: {mean_poison_class_acc}")
    logging.info(f"Mean Attack success rate: {mean_attack_srate}")

    if config['JSON_RESULTS']:
        logging.info("We saved results in json file")
        # saving data inside result_data object, we'll dump it later in a file
        result_data = {}
        result_data['config'] = config
        result_data['mean_poison_class_acc'] = mean_poison_class_acc
        result_data['mean_attack_srate'] = mean_attack_srate
        # result_data['avg_training_losses'] = avg_training_losses
        # result_data['training_losses'] = client_training_losses
        # result_data['testing_losses'] = testing_losses
        result_data['total_accuracies'] = total_accs
        result_data['class_accuracies'] = classes_accs
        

        # storing classwise precision, recall, f1score ans support for every testing round
        # result_data['class_precisions'] = classes_precisions
        # result_data['class_recalls'] = classes_recalls
        result_data['class_f1scores'] = classes_f1scores
        # result_data['class_supports'] = classes_supports
        result_data['avg_metric_vals'] = avg_metric_vals

        # posioned_clients_selected in each round is also stored
        # result_data['poisoned_client_sel'] = poisoned_clients_sel_in_rounds

        # one final test is run and data is saved
        final_test_data = {}
        testing_loss, total_acc, classes_acc, predictions, ground_truths, attack_srate = run_test(server_model, test_data_loader, loss_fn, device)
        final_test_data['poison_class_acc'] = classes_acc[2]
        final_test_data['attack_state'] = attack_srate
        # final_test_data['testing_loss'] = testing_loss
        final_test_data['total_acc'] = total_acc
        final_test_data['classes_acc'] = classes_acc
        # final_test_data['predictions'] = predictions
        # final_test_data['ground_truths'] = ground_truths
        result_data['final_test_data'] = final_test_data

        json_folder = os.path.join(base_path, 'results/json_files/')
        Path(json_folder).mkdir(parents=True, exist_ok=True)
        config_details = f"{config['DATASET']}_C{config['CLIENT_FRAC']}_P{config['POISON_FRAC']}_FDRS{config['FED_ROUNDS']}_CDF{config['COS_DEFENCE']}"
        file_name = '{}_{}.json'.format(config_details, {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())})
        with open(os.path.join(json_folder ,file_name), 'w') as result_file:
            json.dump(result_data, result_file)

    return attack_srates, source_class_accs, total_accs, mean_attack_srate, mean_poison_class_acc
