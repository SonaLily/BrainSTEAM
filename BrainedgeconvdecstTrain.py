from BrainDataset_st import BrainDatasetst
from util.prepossess import mixup_criterion, mixup_data,mixup_data2
import random
import shutil
from sklearn.metrics import roc_curve, auc
import numpy as np
import torch
torch.manual_seed(42)
from torch import nn
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from copy import deepcopy
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def brainedgedecsttrain(model, train_data, criterion, optimizer, n_epochs, batch_size, train_label):

        best_model = deepcopy(model)
        train_best_loss = np.inf
        train_metrics_all = []

        for epoch in range(n_epochs):
            model.train()

            idx_batch = np.random.permutation(int(train_data.shape[0]))
            idx_batch = idx_batch[:int(batch_size)]

            W = train_data.shape[2];
            roi = train_data.shape[3];
            train_data_batch = np.zeros((batch_size, 1, W, roi, 1))
            train_label_batch = train_label[idx_batch]

            for i in range(batch_size):
                r1 = random.randint(0, train_data.shape[2] - W)
                train_data_batch[i] = train_data[idx_batch[i], :, r1:r1 + W, :, :]

            train_data_batch_dev = torch.from_numpy(train_data_batch).float().to(device)
            train_label_batch_dev = torch.from_numpy(train_label_batch).float().to(device)
            train_dataset = BrainDatasetst('ASD_data3.pyg', train_data_batch_dev,train_label_batch_dev)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            shutil.rmtree('ASD_data3.pyg')

            for i, data in enumerate(train_loader, 0):

                mixed_x, mixed_edge_index, y_a, y_b, lam = mixup_data2(data)
                data.x = mixed_x
                data.edge_index = mixed_edge_index
                outputs, x_decoder_1, x_decoder_2, x_decoder_3,degree_truth_1, degree_predict_1, degree_truth_2, degree_predict_2, degree_truth_3, degree_predict_3 = model(data)    #     data)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                degree_truth_1 = degree_truth_1.type(torch.LongTensor)
                degree_truth_2 = degree_truth_2.type(torch.LongTensor)
                degree_truth_3 = degree_truth_3.type(torch.LongTensor)
                degree_predict_1 = degree_predict_1.type(torch.LongTensor)
                degree_predict_2 = degree_predict_2.type(torch.LongTensor)
                degree_predict_3 = degree_predict_3.type(torch.LongTensor)

                loss_alpha = 0.1
                loss_beta = 0.1
                criterion2= nn.MSELoss()

                loss_feature = 0.5*(criterion2(data.x, x_decoder_1)+criterion2(data.x, x_decoder_2)+criterion2(data.x, x_decoder_3))
                loss_degree = 0.5*(criterion(degree_truth_1.float(), degree_predict_1.squeeze()) + criterion(degree_truth_2.float(), degree_predict_2.squeeze())+ criterion(degree_truth_3.float(), degree_predict_3.squeeze()))
                loss = loss + loss_alpha * loss_feature  + loss_beta * loss_degree

                loss.backward()

                optimizer.step()

                optimizer.zero_grad()

            train_result, train_metrics = test_train(model, train_data, batch_size,criterion, train_label,n_epochs)
            train_metrics_all.append(train_metrics)

            print(f'Epoch %i: loss = %f, balanced accuracy = %f'
                  % (epoch, train_metrics['mean_loss'],
                     train_metrics['balanced_accuracy']))

            if train_metrics['mean_loss'] < train_best_loss:
                best_model = deepcopy(model)
                train_best_loss = train_metrics['mean_loss']

        return best_model, train_result, train_metrics_all

def test_test(model, test_data, batch_size, criterion, test_label, epoch=1):

        model.eval()

        columns = ["epoch", "idx", "proba0", "proba1",
                   "true_label", "predicted_label"]
        results_df = pd.DataFrame(columns=columns)
        total_loss = 0

        idx_batch = np.random.permutation(int(test_data.shape[0]))
        idx_batch = idx_batch[:int(batch_size)]

        W = test_data.shape[2];
        roi = test_data.shape[3];
        test_data_batch = np.zeros((batch_size, 1, W, roi, 1))
        test_label_batch = test_label[idx_batch]

        for i in range(batch_size):
            r1 = random.randint(0, test_data.shape[2] - W)
            test_data_batch[i] = test_data[idx_batch[i], :, r1:r1 + W, :, :]

        test_data_batch_dev = torch.from_numpy(test_data_batch).float().to(device)
        test_label_batch_dev = torch.from_numpy(test_label_batch).float().to(device)
        test_dataset = BrainDatasetst('ASD_test_data.pyg', test_data_batch_dev, test_label_batch_dev)
        data_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        shutil.rmtree('ASD_test_data.pyg')

        with torch.no_grad():
            for i, data in enumerate(data_loader, 0):
                labels = data.y

                outputs,_, _, _, _, _, _, _, _, _ = model(data)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                probs = nn.Softmax(dim=1)(outputs)
                _, predicted = torch.max(outputs.data, 1)

                for idx, y in enumerate(data.y):
                    row = [epoch, idx,
                           probs[idx, 0].item(), probs[idx, 1].item(),
                           labels[idx].item(), predicted[idx].item()]
                    row_df = pd.DataFrame([row], columns=columns)
                    results_df = pd.concat([results_df, row_df])

        results_metrics = compute_metrics(results_df.true_label.values, results_df.predicted_label.values)
        results_df.reset_index(inplace=True, drop=True)
        results_metrics['mean_loss'] = total_loss / len(data_loader.dataset)
        results_metrics['epoch'] = epoch

        return results_df, results_metrics

def compute_metrics(ground_truth, prediction):

        tp = np.sum((prediction == 1) & (ground_truth == 1))
        tn = np.sum((prediction == 0) & (ground_truth == 0))
        fp = np.sum((prediction == 1) & (ground_truth == 0))
        fn = np.sum((prediction == 0) & (ground_truth == 1))

        metrics_dict = dict()
        metrics_dict['accuracy'] = (tp + tn) / (tp + tn + fp + fn)

        if tp + fn != 0:
            metrics_dict['sensitivity'] = tp / (tp + fn)
        else:
            metrics_dict['sensitivity'] = 0.0

        if fp + tn != 0:
            metrics_dict['specificity'] = tn / (fp + tn)
        else:
            metrics_dict['specificity'] = 0.0
        metrics_dict['balanced_accuracy'] = (metrics_dict['sensitivity'] + metrics_dict['specificity']) / 2

        metrics_dict['tp'] = tp
        metrics_dict['tn'] = tn
        metrics_dict['fp'] = fp
        metrics_dict['fn'] = fn

        ground_truth = ground_truth.astype(float)
        prediction = prediction.astype(float)
        fpr, tpr, thresholds = roc_curve(prediction, ground_truth)
        roc_auc = auc(fpr, tpr)
        metrics_dict['roc_auc'] = roc_auc

        if tp + fp != 0:
            metrics_dict['precision'] = tp / (tp + fp)
        else:
            metrics_dict['precision'] = 0.0

        if tp + fn != 0:
            metrics_dict['recall'] = tp / (tp + fn)
        else:
            metrics_dict['recall'] = 0.0

        if metrics_dict['precision'] + metrics_dict['recall'] != 0:
            metrics_dict['f1Score'] = 2 * (metrics_dict['precision'] * metrics_dict['recall']) / (
                        metrics_dict['precision'] + metrics_dict['recall'])
        else:
            metrics_dict['f1Score'] = 0.0

        return metrics_dict


def test_train(model, train_data, batch_size, criterion, train_label, epoch=1):

    model.eval()
    columns = ["epoch", "idx", "proba0", "proba1",
               "true_label", "predicted_label"]
    results_df = pd.DataFrame(columns=columns)
    total_loss = 0

    idx_batch = np.random.permutation(int(train_data.shape[0]))
    idx_batch = idx_batch[:int(batch_size)]

    W = train_data.shape[2];
    roi = train_data.shape[3];
    train_data_batch = np.zeros((batch_size, 1, W, roi, 1))
    train_label_batch = train_label[idx_batch]

    for i in range(batch_size):
        r1 = random.randint(0, train_data.shape[2] - W)
        train_data_batch[i] = train_data[idx_batch[i], :, r1:r1 + W, :, :]

    train_data_batch_dev = torch.from_numpy(train_data_batch).float().to(device)
    train_label_batch_dev = torch.from_numpy(train_label_batch).float().to(device)
    train_dataset = BrainDatasetst('ASD_data3.pyg', train_data_batch_dev, train_label_batch_dev)
    data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    shutil.rmtree('ASD_data3.pyg')
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            labels = data.y

            outputs, _, _, _, _, _, _, _, _, _ = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = nn.Softmax(dim=1)(outputs)
            _, predicted = torch.max(outputs.data, 1)

            for idx, y in enumerate(data.y):
                row = [epoch, idx,
                       probs[idx, 0].item(), probs[idx, 1].item(),
                       labels[idx].item(), predicted[idx].item()]
                row_df = pd.DataFrame([row], columns=columns)
                results_df = pd.concat([results_df, row_df])

    results_metrics = compute_metrics(results_df.true_label.values, results_df.predicted_label.values)
    results_df.reset_index(inplace=True, drop=True)
    results_metrics['mean_loss'] = total_loss / len(data_loader.dataset)
    results_metrics['epoch'] = epoch

    return results_df, results_metrics

def compute_metrics(ground_truth, prediction):

    tp = np.sum((prediction == 1) & (ground_truth == 1))
    tn = np.sum((prediction == 0) & (ground_truth == 0))
    fp = np.sum((prediction == 1) & (ground_truth == 0))
    fn = np.sum((prediction == 0) & (ground_truth == 1))

    metrics_dict = dict()
    metrics_dict['accuracy'] = (tp + tn) / (tp + tn + fp + fn)

    if tp + fn != 0:
        metrics_dict['sensitivity'] = tp / (tp + fn)
    else:
        metrics_dict['sensitivity'] = 0.0

    if fp + tn != 0:
        metrics_dict['specificity'] = tn / (fp + tn)
    else:
        metrics_dict['specificity'] = 0.0
    metrics_dict['balanced_accuracy'] = (metrics_dict['sensitivity'] + metrics_dict['specificity']) / 2

    metrics_dict['tp'] = tp
    metrics_dict['tn'] = tn
    metrics_dict['fp'] = fp
    metrics_dict['fn'] = fn

    ground_truth = ground_truth.astype(float)
    prediction = prediction.astype(float)
    fpr, tpr, thresholds = roc_curve(prediction, ground_truth)
    roc_auc = auc(fpr, tpr)
    metrics_dict['roc_auc'] = roc_auc


    if tp + fp != 0:
        metrics_dict['precision'] = tp / (tp + fp)
    else:
        metrics_dict['precision'] = 0.0


    if tp + fn != 0:
        metrics_dict['recall'] = tp / (tp + fn)
    else:
        metrics_dict['recall'] = 0.0

    if metrics_dict['precision'] + metrics_dict['recall'] != 0:
        metrics_dict['f1Score'] = 2 * (metrics_dict['precision'] * metrics_dict['recall']) / (
                metrics_dict['precision'] + metrics_dict['recall'])
    else:
        metrics_dict['f1Score'] = 0.0

    return metrics_dict

def printcsv(FinalTestAccuracy,learning_rate,n_epochs,batch_size,train_metrics_df,test_metrics_df,test_results_df,starttime):
        from datetime import datetime
        from pytz import timezone
        import pytz

        now = datetime.now()
        my_timezone = timezone('US/Mountain')
        starttime=starttime

        now = my_timezone.localize(now)

        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        endtime = date_time
        import time
        timestr = time.strftime("%Y%m%d-%H%M%S")
        FinalTestAccuracy_str = str(FinalTestAccuracy)

        filePath = 'data/results/'
        extension = ".csv"

        program_name = 'main_abide_edgeconv_mixup_dec_st'
        program_path = '/gnnxf_abide_st_Good'
        data_model = 'abide, Braindataset,mixup,DynamicEdgeCovdec,decoder, spactial-temporal'

        output_file = filePath + FinalTestAccuracy_str + program_name + timestr + extension

        learning_rate_str = str(learning_rate)
        n_epochs_str = str(n_epochs)
        batch_size_str = str(batch_size)


        Program_info = [program_name, program_path, data_model, date_time]
        Hyperparameter_info = [learning_rate_str, n_epochs_str, batch_size_str, date_time]

        with open(output_file, 'w') as f:
            f.write('\n'.join(Program_info))
            f.write('\n')
            f.write('start time: ')
            f.write(starttime)
            f.write('\n')
            f.write('end time: ')
            f.write(endtime)
            f.write('\n'.join(Hyperparameter_info))
            f.write('\n')

            f.write('Prediction Performance Metrics:')
            f.write('\n')

        with open(output_file, 'a') as f:
            f.write('\n')
            f.write('Prediction Results:')
            f.write('\n')

        train_metrics_df.to_csv(output_file, mode='a')
        test_metrics_df.to_csv(output_file, mode='a')
        test_results_df.to_csv(output_file, mode='a')


