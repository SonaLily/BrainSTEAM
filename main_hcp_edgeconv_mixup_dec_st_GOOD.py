import yaml
from BrainedgeconvdecstTrain import *
from datetime import datetime
import numpy as np
import torch

torch.manual_seed(42)

import pandas as pd




def main(args):
   
    with open(args.config_filename) as f:

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        learning_rate = 10 ** -4
        n_epochs = 30000 #700 t58 # 200 t62 # 500  t59#150  train 83, test 64,k=10
        batch_size = 32  # 64 -t58 32 -t64 -t57 62

        model = DynamicGNNdec(batch_size)

        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        criterion = torch.nn.NLLLoss()

        for window_size in [128]:  # [50, 64, 75, 100, 128, 256, 1200]:
            W = window_size
            for fold in range(1, 6):
                print('-' * 80)
                print("Window Size {}, Fold {}".format(W, fold))
                print('-' * 80)
                train_data = np.load(
                    '/hcp22/train_data_' + str(fold) + '.npy')
                train_label = np.load(
                    '/hcp22/train_label_' + str(fold) + '.npy')
                test_data = np.load(
                    '/hcp22/test_data_' + str(fold) + '.npy')
                test_label = np.load(
                    '/hcp22/test_label_' + str(fold) + '.npy')
                best_model, train_result, train_metrics = brainedgedecsttrain(model, train_data, criterion, optimizer,n_epochs,batch_size,train_label,learning_rate,test_data,test_label)
                train_results_df, train_metrics = test_train(best_model, train_data, batch_size, criterion,train_label)
                test_results_df, test_metrics = test_test(best_model, test_data,batch_size, criterion,test_label )


        print(test_results_df)
        print(test_metrics)
        train_metrics_df = pd.DataFrame.from_dict([train_metrics])
        test_metrics_df = pd.DataFrame.from_dict([test_metrics])
        FinalTestAccuracyfinal = test_metrics_df['accuracy'] * 100
        FinalTestAccuracyfinal = int(FinalTestAccuracyfinal)

        printcsv(FinalTestAccuracyfinal,learning_rate,n_epochs,batch_size,train_metrics_df,test_metrics_df,test_results_df)

        modelPath = '/data/models/'
        extension = ".pt"

        FinalTestAccuracy = test_metrics_df['accuracy'] * 100
        FinalTestAccuracy = int(FinalTestAccuracy)
        FinalTestAccuracy_str=str(FinalTestAccuracy)
        import time
        timestr = time.strftime("%Y%m%d-%H%M%S")
        Model_file = modelPath + FinalTestAccuracy_str + '_4hcp_dedgeconvk10_Mixup_decoder_st' + timestr + extension

        torch.save(best_model.state_dict(), Model_file)
