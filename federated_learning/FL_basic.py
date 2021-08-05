import copy
from utils import find_indicative_grads
from operator import mod
import time
import os
import logging
import numpy as np
import pandas as pd
from numpy.random import default_rng
from pathlib import Path


from prepare_data import create_client_data, create_client_data_loaders, get_test_data_loader
from available_models import get_model
import json
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

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
aggregate_grads = None

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



def set_initial_trust_vec(dist_type="manual"):
    if dist_type == "manual":
        trust_vec = np.ones((100), dtype=float)
    elif dist_type == "random":
        trust_vec = np.random.random(100)
    elif dist_type == "uniform":
        trust_vec = np.random.uniform(0.0, 1.0, 100)
    else:
        mu, sigma = 0.5, 0.1 # mean and standard deviation
        trust_vec = np.random.normal(mu, sigma, 100)

    trust_vec /= trust_vec.sum()
    logging.info("Initial Trust vector set by method")
    logging.info(trust_vec)
    return trust_vec

def set_initial_trust_mat(dist_type="manual"):
    normalized_local_trusts = list()
    if dist_type == "manual":
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


def init_validation_clients(total_clients, poisoned_clients, rng):
    global config
    # selecting validation clients
    validation_clients_available = [client for client in range(total_clients) if client not in poisoned_clients]
    validation_clients = rng.choice(validation_clients_available, size=int(config['GAMMA'] * total_clients * config['CLIENT_FRAC']), replace=False)
    
    # validation client's trust distribution
    additional_trust_vector = np.zeros(total_clients)
    # replaces the zeros with 1 amount of trust value given to validation client
    np.put(additional_trust_vector, validation_clients, 1/len(validation_clients))

    # we add this additional trust vector to the initial system trust so that system has more trust on validation clients
    # regardless of the trust distribution we use during initialization 
    global current_system_trust_vec
    print("Before trust addition")
    print(current_system_trust_vec.shape)
    print(additional_trust_vector.shape)

    current_system_trust_vec += additional_trust_vector
    current_system_trust_vec /= current_system_trust_vec.sum()

    print("After Validation client trust addition")
    print(current_system_trust_vec)

def eigen_trust(alpha=0.8, epsilon=0.000000001):
    # print("In Eigen Trust")
    ## need to check their dimensions but for now we make it so that they can be multiplied
    global current_system_trust_mat
    global current_system_trust_vec
    matrix_M = np.copy(current_system_trust_mat)

    pre_trust = np.copy(current_system_trust_vec)
    # print(pre_trust)
    
    n = 1000
    new_trust = pre_trust
    # delta_diffs = list()
    for i in range(n):
        old_trust = new_trust
        new_trust = np.matmul(matrix_M.transpose(), old_trust)
        new_trust = alpha*pre_trust + (1-alpha)*new_trust 
        
        dist = np.linalg.norm(new_trust-old_trust)
        # delta_diffs.append(dist)
        if dist <= epsilon:
            print(f"\nno change in trust values after iter: {i} with alpha: {alpha}")
            # print(new_trust)
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


def cos_defence(client_models, computing_clients, poisoned_clients, threshold=0.0):
    ## here poisoned clients are just used for analytics purpose
    global config
    
    ## for trust plots we collect this data
    global all_trust_vals
    global all_client_types
    global client_ids
    global validation_client_ids

    global current_system_trust_mat
    global current_system_trust_vec

    ## first identify validation clients from computing clients, we assume half of the computing clients as
    ## validation clients, top k values
    system_trust_vals = np.array(current_system_trust_vec)
    selected_client_trust_vals = system_trust_vals[computing_clients]
    val_client_num = int(len(computing_clients)/2)
    validating_clients = computing_clients[np.argsort(selected_client_trust_vals)[-val_client_num:]]
    print(f"Validating client selected: {validating_clients}")
    

    ## update trust matrix
    # Step 4, 5 Computing trust and updating the system trust matrix,
    new_system_trust_mat = current_system_trust_mat.copy()
    if config['COLLAB_MODE']:
        global aggregate_grads

        agg_val_vector = torch.zeros(aggregate_grads[validating_clients[0]].size())
        ## join grad vector from all validation client and use that for cosine similarity for all computing clients
        for val_client in validating_clients:
            agg_val_vector += aggregate_grads[val_client]
        
        agg_val_vector = agg_val_vector.reshape(1, -1)
        agg_val_vector /= np.linalg.norm(agg_val_vector)
        
        ## now we iterate over the computing clients to give them trust values
        comp_trusts = list()
        for comp_client in computing_clients:
            comp_vec = copy.deepcopy(aggregate_grads[comp_client]).reshape(1, -1)
            comp_vec /= np.linalg.norm(comp_vec)
            comp_trusts.append((1+cosine_similarity(comp_vec, agg_val_vector)[0][0])/200)
        
        for val_client, comp_client, new_trust_val in zip(validating_clients, computing_clients, comp_trusts):
            prev_val = new_system_trust_mat[val_client, comp_client]
            new_system_trust_mat[val_client, comp_client] = prev_val*config['BETA'] + (1-config['BETA'])*new_trust_val
            
            ## for analytics purposes
            all_trust_vals.append(new_trust_val)
            client_ids.append(comp_client)
            if comp_client in poisoned_clients:
                client_type = 1
                if comp_client//2 == 2:
                    client_type = 2
            else:
                client_type = 0
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

                    prev_val = new_system_trust_mat[val_client, comp_client]
                    new_system_trust_mat[val_client, comp_client] = prev_val*config['BETA'] + (1-config['BETA'])*new_trust_val
                    
                    ## for analytics purposes
                    all_trust_vals.append(new_trust_val)
                    client_ids.append(comp_client)
                    validation_client_ids.append(val_client)
                    if comp_client in poisoned_clients:
                        client_type = 1
                        if comp_client//2 == 2:
                            client_type = 2
                    else:
                        client_type = 0
                    all_client_types.append(client_type) 
            

        ## we need to normalize the new_system_trust_mat row wise
        new_system_trust_mat = new_system_trust_mat / new_system_trust_mat.max(axis=0)
        current_system_trust_mat = new_system_trust_mat

        # Step 6
        # Now we calculate new system trust vector with the help of eigen_trust, since the variables are global
        # we update directly to the current_system_trust_vec
        eigen_trust(alpha=0.4)

        # now aggregate the new global model using this system trust vec
        # in normal_scenario all clients get equal weight but we have given trust vector based on
        # the computation.
        # Thresholding: If we want to set a threhold for minimum trust in order to taken into model averaging
        ## i.e. setting all trust values below threshold = 0.01 to zero.
        current_system_trust_vec = np.where(current_system_trust_vec > threshold, current_system_trust_vec, 0.0)



def train_on_client(idx, model, data_loader, optimizer, loss_fn, device):
    global config
    model.train()
    epoch_training_losses = []

    epoch_grad_bank = dict()
    # epoch_grad_bank['fc1.weight'] = torch.zeros(model.fc1.weight.size())
    # epoch_grad_bank['fc1.bias'] = torch.zeros(model.fc1.bias.size())
    epoch_grad_bank['output_layer.weight'] = torch.zeros(model.output_layer.weight.size())
    epoch_grad_bank['output_layer.bias'] = torch.zeros(model.output_layer.bias.size())


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
            # epoch_grad_bank['fc1.weight'] += model.fc1.weight.grad.detach().clone().cpu()
            # epoch_grad_bank['fc1.bias'] += model.fc1.bias.grad.detach().clone().cpu()
            epoch_grad_bank['output_layer.weight'] += model.output_layer.weight.grad.detach().clone().cpu()
            epoch_grad_bank['output_layer.bias'] += model.output_layer.bias.grad.detach().clone().cpu()


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
    if save_for_feature_finding:
        for key in epoch_grad_bank.keys():
            grad_bank[key].append(epoch_grad_bank[key]/ config['LOCAL_EPOCHS'])
    elif collect_features:
        ## save here what you want to access later for similarity calculation, for e.g. last layer params of the model
        ## or calculated by clustering method to detect important features

        global aggregate_grads
        global indicative_grads
        epoch_grad_vecs = []
        for key in epoch_grad_bank.keys():
            layer_grads = (epoch_grad_bank[key]/config['LOCAL_EPOCHS']).numpy()
            # print("Printing shapes")
            # print(layer_grads.size())
            # print(indicative_grads[key].shape)
            epoch_grad_vecs.append(layer_grads[indicative_grads[key].astype(bool)])
        
        # we just add new params to old ones, during similarity calculation we anyway normalize the whole vector
        # grad_vec = 
        aggregate_grads[idx] += np.concatenate(epoch_grad_vecs).flatten()


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
    # average test loss
    test_loss = test_loss / len(test_data_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

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
        print(cls_acc)
        all_classes_acc.append(cls_acc)
    total_acc = 100. * np.sum(class_correct) / np.sum(class_total)
    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)\n\n' % (
        total_acc, np.sum(class_correct), np.sum(class_total)))

    return test_loss, total_acc, all_classes_acc, predictions, ground_truths


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
    print(f"trust1: {trust1/count}, trust2: {trust2/count} and trust12: {trust12/count}")


def gen_trust_plots(client_ids, validation_client_ids, trust_vals, labels):
    global config
    global base_path
    save_location = os.path.join(base_path, 'results/plots/')
    current_time = time.localtime()
    config_details = f"{config['DATASET']}_C{config['CLIENT_FRAC']}_P{config['POISON_FRAC']}_FDRS{config['FED_ROUNDS']}_CDF{config['COS_DEFENCE']}_CLB{config['COLLAB_MODE']}_LYRS{config['CONSIDER_LAYERS']}_AUROR{config['USE_AUROR']}_CSEP{config['CLUSTER_SEP']}_APLUS{config['USE_AUROR_PLUS']}"

    if config['COLLAB_MODE']:
        ## since in COLLAB_MODE multiple validation clients give trust value we don't have 1:1 ref for
        ## computing client who gave them trust value
        trust_data ={'client_id': client_ids, 'trust_val': trust_vals, 'client_label': labels}
    else:
        trust_data ={'client_id': client_ids, 'validation_client_id': validation_client_ids, 'trust_val': trust_vals, 'client_label': labels}
    
    trust_df = pd.DataFrame.from_dict(trust_data)
    trust_df['modified_trust'] = trust_df['trust_val'].apply(lambda x: int(x*10000))


    ## 1 D Data strip
    strip_fig = px.strip(trust_df, x="modified_trust", y="client_label")
    strip_fig.update_layout(title='Trust given by validation clients, 0-> honest, 1-> minor offender, 2-> major offender')
    strip_fig.write_html(os.path.join(save_location,'{}_trust_strip_{}.html'.format(config_details, time.strftime("%Y-%m-%d %H:%M:%S", current_time))))

    ## histogram of trust vals
    histo_fig = px.histogram(trust_df, x="modified_trust", color="client_label")
    histo_fig.update_layout(title='Trust given by validation clients, 0-> honest, 1-> minor offender, 2-> major offender', barmode="group")
    histo_fig.write_html(os.path.join(save_location,'{}_trust_histo_{}.html'.format(config_details, time.strftime("%Y-%m-%d %H:%M:%S", current_time))))


def trust_clustering(trust_vals, labels):
    trust_arr = np.array(trust_vals).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(trust_arr)
    print(kmeans.cluster_centers_)
    

def start_fl(with_config):
    global config
    config = with_config
    ## config short summary
    config_ss = f"{config['DATASET']}_C{config['CLIENT_FRAC']}_P{config['POISON_FRAC']}_FDRS{config['FED_ROUNDS']}_{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    
    global base_path
    logs_folder = os.path.join(base_path, 'logs/')
    logs_file = logs_folder + config_ss +'.log'
    logging.basicConfig(filename=logs_file, level=getattr(logging, config['LOG_LEVEL']))



    ### initializing global variables block start ###
    global current_system_trust_vec
    global current_system_trust_mat

    ##global variables !! check when running multiple experiments together
    current_system_trust_vec = set_initial_trust_vec("manual")

    ## inital system trust need to be set using three type of distributions
    current_system_trust_mat = set_initial_trust_mat("manual")

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

    ## initializing global grad bank, based on model and layers selected
    layer_names = ['output_layer']
    for name in layer_names:
        grad_bank[name + '.weight'] = list()
        grad_bank[name + '.bias'] = list()

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computing Device:{device}")
    seed = 43
    rng = default_rng(seed)
    # rng = default_rng()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    # If this flag is set first client data is created
    if config['CREATE_DATASET']:
        create_client_data(config['DATASET'], config['CLASS_RATIO'])

    if config['GRAD_COLLECT_FOR'] == -1:
        ## -1 here tells that cos_defence should be started based on the dataset
        if config['DATASET'] == 'mnist':
            start_cosdefence = config['GRAD_COLLECTION_START'] + int(1/config['CLIENT_FRAC'])
        elif config['DATASET'] == 'fmnist':
            start_cosdefence = config['GRAD_COLLECTION_START'] + 2*int(1/config['CLIENT_FRAC'])
        else:
            start_cosdefence = config['GRAD_COLLECTION_START'] + 4 * int(1/config['CLIENT_FRAC'])
    else:
        start_cosdefence = config['GRAD_COLLECTION_START'] + config['GRAD_COLLECT_FOR']


    ## this will return model based on selection
    server_model = get_model(config['MODEL'])
    

    # using gpu for computations if available
    server_model = server_model.to(device)

    # specify loss function (categorical cross-entropy)
    loss_fn = nn.CrossEntropyLoss()

    # choose how many clients you want send model to
    total_clients = 100
    client_models = [copy.deepcopy(server_model).to(device) for _idx in range(total_clients)]

    # location of data with the given config
    data_folder = os.path.join(base_path, f"data/{config['DATASET']}/fed_data/label_flip0/poisoned_{int(config['POISON_FRAC']*100)}CLs/")

    # specify learning rate to be used
    learning_rate = config['LEARNING_RATE']  # change this according to our model, tranfer learning use 0.001, basic model use 0.01
    if config['OPTIMIZER'] == 'sgd':
        optimizers = [optim.SGD(params=client_models[idx].parameters(), lr=learning_rate) for idx in range(total_clients)]
    else:
        optimizers = [optim.Adam(params=client_models[idx].parameters(), lr=learning_rate) for idx in range(total_clients)]
    
    client_data_loaders = create_client_data_loaders(total_clients, data_folder, config['BATCH_SIZE'])
    test_data_loader = get_test_data_loader(config['DATASET'], config['BATCH_SIZE'])

    # Poisoned clients in this setting
    poisoned_clients = []
    poison_config_file = data_folder + 'poison_config.txt'
    with open(poison_config_file, 'r') as pconfig_file:
        pinfo_data = json.load(pconfig_file)
        poisoned_clients = pinfo_data['poisoned_clients']

    if config['COS_DEFENCE']:
        init_validation_clients(total_clients, poisoned_clients, rng)

    ###Training and testing model every kth round
    testing_every = config['TEST_EVERY']
    testing_losses = []
    total_accs = []
    classes_accs = []
    classes_precisions = []
    classes_recalls = []
    classes_f1scores = []
    classes_supports = []
    avg_metric_vals = []
    poisoned_clients_sel_in_round = []
    client_training_losses = [[] for i in range(total_clients)]
    avg_training_losses = [] # this saves the avg loss of the clients selected in one federated round

    ###
    ### Actual federated learning starts here
    ###
    for i in range(config['FED_ROUNDS']):

        ## selecting clients based on probability or always choose clients with highest trust
        print(f"System trust vec sum: {current_system_trust_vec.sum()}")
        # ## Don't know how it can be greater than one here, but to avoid probability error
        # current_system_trust_vec /= current_system_trust_vec.sum()
        if config['SEL_PROB']:
            clients_selected = rng.choice(total_clients, p=current_system_trust_vec, size=int(total_clients * config['CLIENT_FRAC']), replace=False)
        else:
            top_trust_indices = np.argsort(current_system_trust_vec)[-(int(total_clients * config['CLIENT_FRAC'])):]
            ## since out client ids are also numbered from 0 to 99
            clients_selected = top_trust_indices

        print(f"selected clients in round {i}: {clients_selected}")

        poisoned_clients_selected = list(set(poisoned_clients) & set(clients_selected))
        
        print(f"poisoned clients in round {i}: {poisoned_clients_selected}")
        poisoned_clients_sel_in_round.append(len(poisoned_clients_selected))

        temp_training_losses = []
        for j in clients_selected:
            training_losses = train_on_client(j, client_models[j], client_data_loaders[j], optimizers[j], loss_fn, device)
            client_training_losses[j].extend(training_losses)
            temp_training_losses.append(sum(training_losses)/config['LOCAL_EPOCHS'])

        avg_training_losses.append(sum(temp_training_losses)/len(clients_selected))

        # if turned on we change the client_weights from normal to computed by CosDefence
        print(f"CosDefence is On: {config['COS_DEFENCE']}")
        if config['COS_DEFENCE']:
            if i == config['GRAD_COLLECTION_START']:
                save_for_feature_finding = True
            elif i == start_cosdefence - 1:
                indicative_grads, counts = find_indicative_grads(grad_bank, config['CLUSTER_SEP'])
                save_for_feature_finding = False
                collect_features = True
                
                ## this code is upload pre-calculated grad features.
                # layer_names = ['fc1', 'fc2', 'output_layer']
                # counts = 0
                # for name in layer_names:
                #     bias_arr = np.load(name + '.bias.npy')
                #     weight_arr = np.load(name + '.weight.npy')
                #     print(f"Indicative grad of {name} has sizes")
                #     print(bias_arr.shape)
                #     print(weight_arr.shape)
                #     indicative_grads[name + '.bias'] = bias_arr
                #     indicative_grads[name + '.weight'] = weight_arr
                #     counts += np.count_nonzero(bias_arr)
                #     counts += np.count_nonzero(weight_arr)
                
                ## initializing aggregate grads so that now these grads can ve collected as flat vector
                for k in range(total_clients):
                    aggregate_grads.append(torch.zeros(counts))

                print(f"Found {counts} indicative grads")

            elif i >= start_cosdefence:
                cos_defence(client_models, clients_selected, poisoned_clients_selected, 0)


        ## Earlier weight setting strategy
        # client_weights = [1 / (len(clients_selected)) for i in range(total_clients)]  # need to check about this
        # client_weights = torch.tensor(client_weights)

        # ## New weight setting strategy, if cos_defence is on then it modifies current_system_trust_vec, meaning
        # ## it changes the weights of the client selected, if not initial trust vec will be used.
        if config['COS_DEFENCE'] and i >= start_cosdefence:
            client_weights = np.copy(current_system_trust_vec)
            client_weights = torch.from_numpy(client_weights)
            
            ## due to different type of initialization client weights remain low
            ## to correct this we renormalize the weights of the selected clients, so that their sum would be 1
            weights = np.zeros(len(clients_selected), dtype=float)
            for j, client in enumerate(clients_selected):
                weights[j] = client_weights[client]
            print(f"Complete Trust vec {current_system_trust_vec}")
            print(f"Trust value on computing cleints{current_system_trust_vec[clients_selected]}")
            print(weights)
            weights = weights/weights.sum()

            for j, client in enumerate(clients_selected):
                client_weights[client] = weights[j]
            client_weights = client_weights.to(device)
        else:
            ## Earlier weight setting strategy
            client_weights = [1 / (len(clients_selected)) for i in range(total_clients)]  # need to check about this
            client_weights = torch.tensor(client_weights)

        # aggregate to update server_model and client_models
        server_model, client_models = fed_avg(server_model, clients_selected, client_models, client_weights)
        print(f"Round {i} complete")
    
        # Testing Model every kth round
        if (i + 1) % testing_every == 0:
            testing_loss, total_acc, classes_acc, predictions, ground_truths = run_test(server_model, test_data_loader, loss_fn, device)
            cls_precisions, cls_recalls, cls_f1scores, cls_supports = metrics.precision_recall_fscore_support(ground_truths, predictions, average=None, zero_division=1)
            classes_precisions.append(cls_precisions.tolist())
            classes_recalls.append(cls_recalls.tolist())
            classes_f1scores.append(cls_f1scores.tolist())
            classes_supports.append(cls_supports.tolist())
            # we store "weighted" average values of precision, recall, f1score and support in this list.
            avg_metric_vals.append(metrics.precision_recall_fscore_support(ground_truths, predictions, average='weighted', zero_division=1))
            testing_losses.append(testing_loss)
            total_accs.append(total_acc)
            classes_accs.append(classes_acc)

    if config['COS_DEFENCE']:
        gen_trust_plots(client_ids, validation_client_ids, all_trust_vals, all_client_types)
        trust_clustering(all_trust_vals, all_client_types)

    if config['JSON_RESULTS']:
        print("We saved results in json file")
        # saving data inside result_data object, we'll dump it later in a file
        result_data = {}
        result_data['config'] = config
        result_data['avg_training_losses'] = avg_training_losses
        result_data['training_losses'] = client_training_losses
        result_data['testing_losses'] = testing_losses
        result_data['total_accuracies'] = total_accs
        result_data['class_accuracies'] = classes_accs

        # storing classwise precision, recall, f1score ans support for every testing round
        result_data['class_precisions'] = classes_precisions
        result_data['class_recalls'] = classes_recalls
        result_data['class_f1scores'] = classes_f1scores
        result_data['class_supports'] = classes_supports
        result_data['avg_metric_vals'] = avg_metric_vals

        # posioned_clients_selected in each round is also stored
        result_data['poisoned_client_sel'] = poisoned_clients_sel_in_round

        # one final test is run and data is saved
        final_test_data = {}
        testing_loss, total_acc, classes_acc, predictions, ground_truths = run_test(server_model, test_data_loader, loss_fn, device)
        final_test_data['testing_loss'] = testing_loss
        final_test_data['total_acc'] = total_acc
        final_test_data['classes_acc'] = classes_acc
        final_test_data['predictions'] = predictions
        final_test_data['ground_truths'] = ground_truths
        result_data['final_test_data'] = final_test_data

        json_folder = os.path.join(base_path, 'results/json_files/')
        Path(json_folder).mkdir(parents=True, exist_ok=True)
        config_details = f"{config['DATASET']}_C{config['CLIENT_FRAC']}_P{config['POISON_FRAC']}_FDRS{config['FED_ROUNDS']}_LAYERS{config['CONSIDER_LAYERS']}_AUROR{config['USE_AUROR']}_CSEP{config['CLUSTER_SEP']}_APLUS{config['USE_AUROR_PLUS']}"
        file_name = '{}_{}.txt'.format(config_details, {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())})
        with open(os.path.join(json_folder ,file_name), 'w') as result_file:
            json.dump(result_data, result_file)
