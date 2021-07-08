import copy
from operator import mod
import time
import os
import numpy as np
from numpy.random import default_rng, gamma, normal
import argparse
from pathlib import Path
from prepare_data import create_client_data, create_client_data_loaders, get_test_data_loader
from available_models import BasicFCN, BasicCNN
import json
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import torch.optim as optim


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
    elif dist_type == "dirichlet":
        pass
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
    print("Initial Trust Mat set by method")
    print(trust_mat)
    return trust_mat


def set_initial_trust_vec(dist_type="manual"):
    if dist_type == "manual":
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

    trust_vec = trust_vec/trust_vec.sum()
    print("Initial Trust vector set by method")
    print(trust_vec)
    return trust_vec


##global variables !! check when running multiple experiments together
current_system_trust_vec = set_initial_trust_vec("manual")

## inital system trust need to be set using three type of distributions
current_system_trust_mat = set_initial_trust_mat("manual")

## global grads, we will initialize it again based on the number of parameters it stores
## e.g. last layer of the model
aggregate_grads = []

## To check the difference in trust values
single_malicious_client_diffs = []
both_honest_client_diffs = []
both_malicious_client_diffs = []
single_malicious_trust_vals = []
both_honest_client_trust_vals = []
both_malicious_client_trust_vals = []


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

    current_system_trust_vec = new_trust


def get_validation_clients(poisoned_clients, client_frac, total_clients, rng, gamma=1.0):
    # selecting validation clients
    validation_clients_available = [client for client in range(100) if client not in poisoned_clients]
    validation_clients = rng.choice(validation_clients_available, size=int(gamma * total_clients * client_frac), replace=False)
    
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
    
    return validation_clients

def identify_poisoned(clients_selected, poisoned_clients):
    posioned_client_selected = []
    for client in poisoned_clients:
        if client in clients_selected:
            posioned_client_selected.append(client)

    return posioned_client_selected

def validate_computation(val_model, val_id, comp1_model, comp1_id, comp2_model, comp2_id):

    ## OPTION1 : last computation params
    # here for now we are just checking similarity between last layer params, can be changed later
    # !! check if tensor accessing this way is okay or we need to make copy
    # comp1_weight_fc2 = comp1_model.state_dict()['fc2.weight']
    # comp1_bias_fc2 = comp1_model.state_dict()['fc2.bias']
    # comp1_weight_op = comp1_model.state_dict()['output_layer.weight']
    # comp1_bias_op = comp1_model.state_dict()['output_layer.bias']
    # comp1_vec = torch.cat([comp1_weight_fc2.reshape(1, -1), comp1_bias_fc2.reshape(1, -1), comp1_weight_op.reshape(1, -1), comp1_bias_op.reshape(1, -1)], axis=1).cpu()
    
    
    # comp2_weight_fc2 = comp2_model.state_dict()['fc2.weight']
    # comp2_bias_fc2 = comp2_model.state_dict()['fc2.bias']
    # comp2_weight_op = comp2_model.state_dict()['output_layer.weight']
    # comp2_bias_op = comp2_model.state_dict()['output_layer.bias']
    # comp2_vec = torch.cat([comp2_weight_fc2.reshape(1, -1), comp2_bias_fc2.reshape(1, -1), comp2_weight_op.reshape(1, -1), comp2_bias_op.reshape(1, -1)], axis=1).cpu()


    # val_weight_fc2 = val_model.state_dict()['fc2.weight']
    # val_bias_fc2 = val_model.state_dict()['fc2.bias']
    # val_weight_op = val_model.state_dict()['output_layer.weight']
    # val_bias_op = val_model.state_dict()['output_layer.bias']
    # val_vec = torch.cat([val_weight_fc2.reshape(1, -1), val_bias_fc2.reshape(1, -1), val_weight_op.reshape(1, -1), val_bias_op.reshape(1, -1)], axis=1).cpu()



    ## use all parameters to make a single vector
    # comp1_vec = []
    # for param in comp1_model.parameters():
    #     comp1_vec.extend(param.data.cpu())
    # comp2_vec = []
    # for param in comp2_model.parameters():
    #     comp2_vec.extend(param.data.cpu())
    
    # val_vec = []
    # for param in val_model.parameters():
    #     val_vec.extend(param.data.cpu())
    
    # comp1_vec = np.array(comp1_vec, dtype=float).reshape(1, -1)
    # comp2_vec = np.array(comp2_vec, dtype=float).reshape(1, -1)
    # val_vec = np.array(val_vec, dtype=float).reshape(1, -1)

    # k = 10
    # trust_comp1 = cosine_similarity(torch.topk(val_vec, k)[0], torch.topk(comp1_vec, k)[0])/100
    # trust_comp2 = cosine_similarity(torch.topk(val_vec, k)[0], torch.topk(comp2_vec, k)[0])/100
    
    ## OPTION2 : Aggregated updates
    global aggregate_grads
    comp1_vec = aggregate_grads[comp1_id]
    comp2_vec = aggregate_grads[comp2_id]
    val_vec = aggregate_grads[val_id]



    ## normalize the vectors
    comp1_vec /= comp1_vec.sum()
    comp2_vec /= comp2_vec.sum()
    val_vec /= val_vec.sum()

    trust_comp1 = cosine_similarity(val_vec, comp1_vec)/100
    trust_comp2 = cosine_similarity(val_vec, comp2_vec)/100
    trust_between_comps = cosine_similarity(comp1_vec, comp2_vec)/100

    return trust_comp1, trust_comp2, trust_between_comps


def cos_defence(client_models, client_data_loaders, optimizers, loss_fn, local_epochs, device, computing_clients, poisoned_clients, validation_clients, overlapping, beta=0.1, gamma=1.0, threshold=0.0):
    # Step 4 Sending updates to validating clients
    # validating client dictionary, stores info of (computing_client1, computing_client2) for validating clients
    validating_info = dict()
    for val_client in validation_clients:
        validating_info[val_client] = list()
    ## implementing simple algorithm without consideration of overlapping
    if overlapping:
        pass
    else:
        num_comp = len(computing_clients)
        num_val = len(validation_clients)
        k = 0
        for client in computing_clients:
            j = 0
            while(j < 2*gamma):
                validating_info[validation_clients[k]].append(client)
                j = j + 1
                k = (k + 1)%num_val
        
        # Step 4, 5 Computing trust and updating the system trust matrix,
        # !! need to think if we should normalize the trust matrix again
        # !! also think if local trust and normalized local trust matrix make sense
        # now we iterate over the dictionary and update the system trust matrix

        # first we train on validation client, new params are saved directly
        for j in validation_clients:
            train_on_client(j, client_models[j], client_data_loaders[j], optimizers[j], loss_fn, local_epochs, device)
        
        ## need to think about whether we need to normalize the whole matrix again?
        global current_system_trust_mat
        new_system_trust_mat = current_system_trust_mat.copy()
        global single_malicious_client_diffs
        global single_malicious_trust_vals
        global both_honest_client_diffs
        global both_honest_client_trust_vals
        global both_malicious_client_diffs
        global both_malicious_client_trust_vals

        # print(new_system_trust_mat)
        for key, val in validating_info.items():
            # print(new_system_trust_mat[key])
            trust_client1, trust_client2, trust_bw_clients = validate_computation(client_models[key], key, client_models[val[0]], val[0], client_models[val[1]], val[1])
            print(f"Trust value given by {key} for {val[0]} is: {trust_client1}, and for {val[1]} is: {trust_client2}")
            if val[0] in poisoned_clients:
                print("client 1 is poisoned")
            if val[1] in poisoned_clients:
                print("client 2 is poisoned")

            diff = abs(trust_client1-trust_client2)
            if (val[0] in poisoned_clients) and (val[1] in poisoned_clients):
                both_malicious_client_diffs.append(*diff)
                both_malicious_client_trust_vals.append((trust_client1, trust_client2, trust_bw_clients))
            elif (val[0] in poisoned_clients) or (val[1] in poisoned_clients):
                single_malicious_client_diffs.append(*diff)

                ## to collect all malicious trust values in one place
                if val[0] in poisoned_clients:
                    single_malicious_trust_vals.append((trust_client1, trust_client2, trust_bw_clients))
                else:
                    single_malicious_trust_vals.append((trust_client2, trust_client1, trust_bw_clients))
            else:
                both_honest_client_diffs.append(*diff)
                both_honest_client_trust_vals.append((trust_client1, trust_client2, trust_bw_clients))
            


            # print(f"previous trust row by this val: {key} client: {new_system_trust_mat[key]}")
            #beta implementation different than proposal reverse meaning, beta = 1.0 full history retention no updating
            prev_val1 = new_system_trust_mat[key, val[0]]
            new_system_trust_mat[key, val[0]] = prev_val1*beta + (1-beta)*trust_client1

            prev_val2 = new_system_trust_mat[key, val[1]]
            new_system_trust_mat[key, val[1]] = prev_val2*beta + (1-beta)*trust_client2

            # print(f"After update, trust row by this val: {key} client: {new_system_trust_mat[key]}")
        
        ## we need to normalize the new_system_trust_mat row wise or columnwise, that needs to be seen, for now rowwise
        for i in range(100):
            new_system_trust_mat[i] = new_system_trust_mat[i]/new_system_trust_mat[i].sum()

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
        global current_system_trust_vec
        current_system_trust_vec = np.where(current_system_trust_vec > threshold, current_system_trust_vec, 0.0)



def train_on_client(idx, model, data_loader, optimizer, loss_fn, local_epochs, device):
    model.train()
    epoch_training_losses = []
    for epoch in range(local_epochs):
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
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()

        epoch_train_loss = train_loss / len(data_loader)
        epoch_training_losses.append(epoch_train_loss)
        # print('Client: {}\t Epoch: {} \tTraining Loss: {:.6f}'.format(idx, epoch, epoch_train_loss))

    ## save here what you want to access later for similarity calculation, for e.g. last layer params of the model
    global aggregate_grads
    model_weight_op = model.state_dict()['output_layer.weight']
    model_bias_op = model.state_dict()['output_layer.bias']
    model_vec = torch.cat([model_weight_op.reshape(1, -1), model_bias_op.reshape(1, -1)], axis=1).cpu()
    # we just add new params to old ones, during similarity calculation we anyway normalize the whole vector 
    aggregate_grads[idx] += model_vec


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


def main():
    parser = argparse.ArgumentParser("To run FL from CLI")

    ## To choose if we want to run experiment with cmd args or config.py, will implment later
    parser.add_argument('--cfg_mode', action='store_true', help="if given the experiment will be run from config file")

    ## FL environment settings
    parser.add_argument('-d_sel', '--dataset_selection', default='mnist', help='mnist|cifar10')
    parser.add_argument('-c_frac', '--client_frac', type=float, default=0.1, help='client fraction to select in each round')
    parser.add_argument('-p_frac', '--poison_frac', type=float, default=0.0, help='poisoned fraction to select 0.0|0.1|0.2|0.4')
    parser.add_argument('-ccds', '--create_cdata', action='store_true', help='create client datasets')

    ## config setting for CosDefence mechanism
    parser.add_argument('-c_def', '--cos_defence', action='store_true', help='to turn on CosDefence mechanism')
    parser.add_argument('--alpha', type=float, default=0.8, help='initial trust importance factor')
    parser.add_argument('--beta', type=float, default=0.7, help='trust history retention factor')
    parser.add_argument('--gamma', type=float, default=1.0, help='redundancy factor')
    parser.add_argument('--val_olp', action='store_true', help='validation client can be a computing client also')
    parser.add_argument('--grad_agg', action='store_true', help='aggregate gradients to compute similarity')
    parser.add_argument('--dynamic', action='store_true', help='after some iterations validation client can be selected dynamically')
    
    ## FL settings
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2, help='learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-opt', '--optimizer', default='Adam', help='select one Adam|SGD')
    parser.add_argument('-fdrs', '--fed_rounds', type=int, default=10, help='iterations for communication')
    parser.add_argument('-tevr', '--testing_every', type=int, default=4, help='testing model after every kth round')
    parser.add_argument('-leps', '--local_epochs', type=int, default=3, help='optimization epochs in local worker between communication')
    
    ## log settings
    parser.add_argument('--log', action='store_true', help='whether to make a log')
    parser.add_argument('--jlog', action='store_true', help='creates a json log file used to later visualize results')
    
    args = parser.parse_args()
    print(args)

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computing Device:{device}")
    seed = 42
    rng = default_rng(seed)
    # rng = default_rng()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    # If this flag is set first client data is created
    if args.create_cdata:
        create_client_data(args.dataset_selection)

    if args.dataset_selection == 'mnist':
        server_model = BasicFCN()
    else:
        server_model = BasicCNN()

    # using gpu for computations if available
    server_model = server_model.to(device)

    # specify loss function (categorical cross-entropy)
    loss_fn = nn.CrossEntropyLoss()

    # choose how many clients you want send model to
    total_clients = 100
    client_models = [copy.deepcopy(server_model).to(device) for _idx in range(total_clients)]

    ## initializing this list also for the shape we need, for later storing params after model training
    global aggregate_grads
    model_weight_op = server_model.state_dict()['output_layer.weight']
    model_bias_op = server_model.state_dict()['output_layer.bias']
    model_vec = torch.cat([model_weight_op.reshape(1, -1), model_bias_op.reshape(1, -1)], axis=1).cpu()
    for i in range(total_clients):
        aggregate_grads.append(torch.zeros(model_vec.shape, dtype=float))

    fed_rounds = args.fed_rounds
    local_epochs = args.local_epochs
    batch_size = args.batch_size

    # location of data with the given config and logs location
    raw_data_folder = os.path.join(base_path, f'data/{args.dataset_selection}/raw_data/')
    logs_folder = os.path.join(base_path, 'logs/')
    data_folder = os.path.join(base_path, f'data/{args.dataset_selection}/fed_data/label_flip0/poisoned_{int(args.poison_frac*100)}CLs/')

    # specify learning rate to be used
    learning_rate = args.learning_rate  # change this according to our model, tranfer learning use 0.001, basic model use 0.01
    if args.optimizer == 'SGD':
        optimizers = [optim.SGD(params=client_models[idx].parameters(), lr=learning_rate) for idx in range(total_clients)]
    else:
        optimizers = [optim.Adam(params=client_models[idx].parameters(), lr=learning_rate) for idx in range(total_clients)]
    
    client_data_loaders = create_client_data_loaders(total_clients, data_folder, batch_size)
    test_data_loader = get_test_data_loader(args.dataset_selection, batch_size)

    # Poisoned clients in this setting
    poisoned_clients_available = []
    poison_config_file = data_folder + 'poison_config.txt'
    with open(poison_config_file, 'r') as pconfig_file:
        pinfo_data = json.load(pconfig_file)
        poisoned_clients_available = pinfo_data['poisoned_clients']

    ###Training and testing model every kth round
    testing_every = args.testing_every
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

    # before running computation we identify validation clients, based on overlapping setting and poisoned clients
    validation_clients = get_validation_clients(poisoned_clients_available, args.client_frac, total_clients, rng, args.gamma)
    print("Validation Client selected:")
    print(validation_clients)
    # for now we keep static setting, meaning we will keep these validation clients fixed
    if args.val_olp:
        computing_clients_available = [client for client in range(100)]
    else:
        computing_clients_available = [client for client in range(100) if client not in validation_clients]

    ###
    ### Actual federated learning starts here
    ###
    for i in range(fed_rounds):
        if args.cos_defence:
            selection_probs = np.zeros(len(computing_clients_available), dtype=float)
            for j, client in enumerate(computing_clients_available):
                selection_probs[j] = current_system_trust_vec[client]
            selection_probs = selection_probs/selection_probs.sum()
            clients_selected = rng.choice(computing_clients_available,p=selection_probs, size=int(total_clients * args.client_frac), replace=False)
        else:
            clients_selected = rng.choice(computing_clients_available, size=int(total_clients * args.client_frac), replace=False)
        
        print(f"selected clients in round {i}: {clients_selected}")

        poisoned_clients = list(set(poisoned_clients_available) & set(clients_selected))
        
        print(f"poisoned clients in round {i}: {poisoned_clients}")
        poisoned_clients_sel_in_round.append(len(poisoned_clients))

        temp_training_losses = []
        for j in clients_selected:
            training_losses = train_on_client(j, client_models[j], client_data_loaders[j], optimizers[j], loss_fn,
                                              local_epochs, device)
            client_training_losses[j].extend(training_losses)
            temp_training_losses.append(sum(training_losses)/local_epochs)
        avg_training_losses.append(sum(temp_training_losses)/len(clients_selected))
        # aggregate to update server_model and client_models
        print(f"Round {i} complete")

        # if turned on we change the client_weights from normal to computed by CosDefence
        print(f"CosDefence is On: {args.cos_defence}")
        if args.cos_defence:
            cos_defence(client_models, client_data_loaders, optimizers, loss_fn, local_epochs, device, clients_selected, poisoned_clients, validation_clients, False)
        
        ## Earlier weight setting strategy
        # client_weights = [1 / (total_clients*args.client_frac) for i in range(total_clients)]  # need to check about this
        # client_weights = torch.tensor(client_weights)

        ## New weight setting strategy
        client_weights = np.copy(current_system_trust_vec)
        client_weights = torch.from_numpy(client_weights)
        ## due to different type of initialization client weights remain low
        ## to correct this we renormalize the weights of the selected clients, so that their sum would be 1
        weights = np.zeros(len(clients_selected), dtype=float)
        for j, client in enumerate(clients_selected):
            weights[j] = client_weights[client]
        weights = weights/weights.sum()
        for j, client in enumerate(clients_selected):
            client_weights[client] = weights[j]
        client_weights = client_weights.to(device)


        server_model, client_models = fed_avg(server_model, clients_selected, client_models, client_weights)
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

    ## after fed rounds, printing trust difference
    global single_malicious_client_diffs
    global single_malicious_trust_vals
    global both_honest_client_diffs
    global both_honest_client_trust_vals
    global both_malicious_client_diffs
    global both_malicious_client_trust_vals
    print("Here are trust diffs")
    # print(single_malicious_client_diffs)
    print(sum(single_malicious_client_diffs)/len(single_malicious_client_diffs))
    # print(both_malicious_client_diffs)
    print(sum(both_malicious_client_diffs)/len(both_malicious_client_diffs))
    # print(both_honest_client_diffs)
    print(sum(both_honest_client_diffs)/len(both_honest_client_diffs))

    print("Raw trust vals")
    print("Singel malicious client (trust1 values are for malicious client)")
    print_trust_vals(single_malicious_trust_vals)
    print("Both malicious clients")
    print_trust_vals(both_malicious_client_trust_vals)
    print("Both honest clients")
    print_trust_vals(both_honest_client_trust_vals)    

    if args.log:
        log_path = os.path.join(base_path, 'logs/')
        Path(log_path).mkdir(parents=True, exist_ok=True)
        current_time = time.localtime()
        logfile = open(os.path.join(log_path, '{}.log'.format(time.strftime("%Y-%m-%d %H:%M:%S", current_time))), 'a')
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", current_time)))
        logfile.write('===Configs===\n')
        logfile.write('d_sel: {}\n'.format(args.dataset_selection))
        logfile.write('c_frac: {}\n'.format(args.client_frac))
        logfile.write('p_frac: {}\n'.format(args.poison_frac))
        logfile.write('lr: {}\n'.format(args.learning_rate))
        logfile.write('bs: {}\n'.format(args.batch_size))
        logfile.write('fdrs: {}\n'.format(args.fed_rounds))
        logfile.write('tevr: {}\n'.format(args.testing_every))
        logfile.write('leps: {}\n'.format(args.local_epochs))

        ###Logging of losses
        logfile.write('\n\n===Training Losses===\n')
        for i in range(args.fed_rounds):
            logfile.write(f'average_training_loss_round_{i+1} : {avg_training_losses[i]}\n')
        # for i in range(total_clients):
        #     for j in range(len(client_training_losses[i])):
        #         logfile.write(f'training_loss_client_{i} epoch: {j + 1}\tloss: {client_training_losses[i][j]}\n')

        ###Logging of Testing losses and testing accs
        logfile.write('\n\n===Testing Loss and Model Accuracy===\n')
        for i in range(len(testing_losses)):
            logfile.write(f'testing loss round: {(i + 1) * testing_every}\tloss: {testing_losses[i]}\n')
            logfile.write(f'Model Acc round: {(i + 1) * testing_every}\tAcc: {total_accs[i]}%\n')
            for j in range(10):
                logfile.write(str(classes_accs[i][j]) +'\n')
            print('\n\n')
        logfile.close()

    if args.jlog:
        print("We save json")
        # saving data inside result_data object, we'll dump it later in a file
        result_data = {}
        config = {
            "client_frac" : args.client_frac,
            "poison_frac" : args.poison_frac,
            "learning_rate" : args.learning_rate,
            "cos_defence"   : args.cos_defence,
            "alpha"         : args.alpha,
            "beta"          : args.beta,
            "gamma"         : args.gamma,
            "val_olp"       : args.val_olp,
            "dynamic"       : args.dynamic,
            "batch_size"    : args.batch_size,
            "optimizer"     : args.optimizer,
            "fed_rounds"    : args.fed_rounds,
            "testing_every" : args.testing_every,
            "local_epochs"  : args.local_epochs
        }
        result_data['data'] = args.dataset_selection
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

        json_folder = os.path.join(base_path, 'json_files/')
        Path(json_folder).mkdir(parents=True, exist_ok=True)
        config_details = f'{args.dataset_selection}_C{args.client_frac}_P{args.poison_frac}_FDRS{args.fed_rounds}_LR{args.learning_rate}_Opt{args.optimizer}'
        file_name = '{}.txt'.format(config_details)
        with open(os.path.join(json_folder ,file_name), 'w') as result_file:
            json.dump(result_data, result_file)



if __name__ == '__main__':
    main()