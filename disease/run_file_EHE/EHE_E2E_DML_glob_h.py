import torch
import math
import numpy as np
import argparse
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from processing.CenteredScaled import CenteredScaled
from EHE_E2E_models.E2E_EHE_RigidNet_D import RigidNet
from EHE_E2E_models.E2E_EHE_NonRigidNet_D import NonRigidNet
np.random.seed(42)

# ** EHE **
for ex in range(1,7):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_runs', default=10,type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--num_epoch', default=35, type=int)
    parser.add_argument('--use_cuda', default=1, type=int)
    opt = parser.parse_args()

    # options=['RigidTransform','NonRigidTransform','RigidTransformInit','NonRigidTransformInit']
    options=['RigidTransform']

    training_epochs= opt.num_epoch
    learning_rate=opt.learning_rate
    runs = opt.num_runs

    use_cuda = 1
    cuda = torch.cuda.is_available()
    device = 'cuda' if (cuda == True and use_cuda == 1) else 'cpu'
    if device == 'cuda':
        print('Using CUDA')
        torch.cuda.empty_cache()
    else:
        print('NOT using CUDA')

    print("==== NOW RUNNING FOR EXERCISE ====", ex)

    folder_path = '../data_and_features/Extracted_Features/cv_cs/xyz/{}'.format(ex)
    folds_acc = {}

    for fold in range(len(os.listdir(folder_path))):

        batch_size = 12
        batch_size_test = 12

        print('Loading data for fold {}'.format(fold+1))
        X_train_pos = np.load(os.path.join(folder_path, str(fold+1), 'train_data.npy'),  allow_pickle=True)
        y_train = np.load(os.path.join(folder_path, str(fold+1), 'train_label.pkl'), allow_pickle=True)
        X_test_pos = np.load(os.path.join(folder_path, str(fold+1), 'eval_data.npy'),  allow_pickle=True)
        y_test = np.load(os.path.join(folder_path, str(fold+1), 'eval_label.pkl'), allow_pickle=True)

        X_train = X_train_pos.transpose(0, 2, 3, 1, 4).squeeze(-1)
        X_test = X_test_pos.transpose(0, 2, 3, 1, 4).squeeze(-1)
        print("--------------------------------------")

        # get labels
        y_train = y_train[1]
        y_test = y_test[1]
        y_train = np.array(y_train).astype('int32')
        y_test = np.array(y_test).astype('int32')
            
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)

        print('Preprocessing (going to preshape space)')

        for i in range(X_train.shape[0]):
            for j in range(X_train[i].shape[0]):
                X_train[i, j] = CenteredScaled(X_train[i, j])

        for i in range(X_test.shape[0]):
            for j in range(X_test[i].shape[0]):
                X_test[i, j] = CenteredScaled(X_test[i, j])

        print('Pre-shape space preprocessing completed')

        num_frames = X_train.shape[1]
        num_joints = X_train.shape[2]
        dims = X_train.shape[3]
        num_channels = num_joints * dims

        acc_ = []
        for m in options:
            loss = []
            mod = m
            run_acc = []
            print('Running {} for fold {}'.format(mod,fold+1))
            for r in range(runs):
                dml_run_acc = []
                # print('Running {} for {} for fold {}'.format(r,mod,fold+1))
                for dml_run in range(runs):
                    if m == 'RigidTransform' or m == 'RigidTransformInit':
                        rigid = True
                    else:
                        rigid = False

                    if rigid:
                        model_class = RigidNet          
                    else:
                        model_class = NonRigidNet
                
                    model = model_class(mod=mod, r=r, dml_run=dml_run, glob_h=True).to(device)
                    criterion = nn.BCEWithLogitsLoss(reduction='sum')
                    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
                    steps = len(X_train) // batch_size
                    # print('model will be training for {} epochs with {} steps per epoch'.format(training_epochs,steps))

                    model.train()
                    for epoch in range(training_epochs):
                        correct=0
                        total=0
                        epoch_loss = 0.0
                        for i in range(steps + (1 if len(X_train) % batch_size != 0 else 0)):                       
                            start_idx = i * batch_size
                            end_idx = start_idx + batch_size
                            x, y = X_train[start_idx:end_idx], y_train[start_idx:end_idx]
                            # If the batch size exceeds the remaining samples, adjust the slicing
                            if end_idx > len(X_train):
                                x = X_train[start_idx:]
                                y = y_train[start_idx:]
                            inputs, labels = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
                            opt.zero_grad()   
                            output = model(inputs.float())
                            loss = criterion(output[:, 0], labels.float())
                            loss.backward()
                            opt.step()
                            epoch_loss += loss.item()
                            y_pred_sigmoid = torch.round(torch.sigmoid(output.data))
                            total += labels.size(0)
                            correct += (y_pred_sigmoid[:,0] == labels.long()).sum().item()

                        accuracy = 100 * correct / total
                        epoch_loss = epoch_loss / len(X_train) 
                        # print("Training Accuracy = {} : Training Loss {}".format(accuracy,epoch_loss))
                    
                    correct_test = 0
                    total_test = 0
                    model.eval()
                    with torch.no_grad():
                        steps = len(X_test) // batch_size_test
                        test_proba = np.zeros(len(X_test))
                        for i in range(steps + (1 if len(X_test) % batch_size_test != 0 else 0)):                 
                            start_idx = i * batch_size_test
                            end_idx = start_idx + batch_size_test
                            x, y = X_test[start_idx:end_idx], y_test[start_idx:end_idx]
                            # If the batch size exceeds the remaining samples, adjust the slicing
                            if end_idx > len(X_test):
                                x = X_test[start_idx:]
                                y = y_test[start_idx:]                    
                            inputs, labels = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
                            outputs = model(inputs.float())
                            y_pred_sigmoid = torch.round(torch.sigmoid(outputs.data))
                            total_test += labels.size(0)
                            correct_test += (y_pred_sigmoid[:,0] == labels.long()).sum().item()
                                    
                    accuracy = 100*correct_test/total_test
                    dml_run_acc.append(accuracy)
                run_acc.append(max(dml_run_acc))
            acc_.append(max(run_acc))
            max_index = run_acc.index(max(run_acc))

        folds_acc[fold+1] = acc_

    print('Accuracies for all folds {} :'.format(folds_acc))

    # *** Compute the mean of each transform on all the folds.
    acc_values = list(folds_acc.values())
    mean = [sum(elements) / len(elements) for elements in zip(*acc_values)]
    print('--------------------------------')
    print('The mean of each transform across five folds is {}'.format(mean))
    print('--------------------------------')
    max_of_mean = max(mean)
    max_of_mean_index = mean.index(max_of_mean)

    if max_of_mean_index == 0 :
        print("The best transform layer for this exercise based on the 5 folds is {}, with mean acc of {}".format(options[max_of_mean_index], max_of_mean))
    elif max_of_mean_index == 1:
        print("The best transform layer for this exercise based on the 5 folds is {}, with mean acc of {}".format(options[max_of_mean_index], max_of_mean))
    elif max_of_mean_index == 2:
        print("The best transform layer for this exercise based on the 5 folds is {}, with mean acc of {}".format(options[max_of_mean_index], max_of_mean))
    else:
        print("The best transform layer for this exercise based on the 5 folds is {}, with mean acc of {}".format(options[max_of_mean_index], max_of_mean))

