
import yaml
from BrainedgeconvdecstTrain import *
from datetime import datetime
import numpy as np


import torch
torch.manual_seed(42)


import pandas as pd




def main(args):
   
    with open(args.config_filename) as f:
        config = yaml.load(f, Loader=yaml.Loader)



        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)


        loss_name = 'loss'
        now = datetime.now()

        date_time = now.strftime("%m-%d-%H-%M-%S")
        starttime=date_time



        extractor_type = config['model']['extractor_type'] if 'extractor_type' in config['model'] else "none"
        embedding_size = config['model']['embedding_size'] if 'embedding_size' in config['model'] else "none"
        window_size = config['model']['window_size'] if 'window_size' in config['model'] else "none"

        if "graph_generation" in config['model'] and config['model']["graph_generation"]:
            model_name = f"{config['train']['method']}_{config['model']['graph_generation']}"
        else:
            model_name = f"{config['train']['method']}"

        save_folder_name = Path(config['train']['log_folder'])/Path(
            date_time +
            f"_{config['data']['dataset']}_{config['model']['type']}_{model_name}"
            + f"_{extractor_type}_{loss_name}_{embedding_size}_{window_size}")



        learning_rate = 10 ** -4
        n_epochs = 10000
        batch_size = 32

        model = DynamicGNNdec(batch_size)



        optimizer = torch.optim.Adam(model.parameters(), learning_rate)

        criterion = torch.nn.NLLLoss()



        for window_size in [128]:
            W = window_size

            for fold in range(1, 5):
                print('-' * 80)
                print("Window Size {}, Fold {}".format(W, fold))
                print('-' * 80)


                train_data = np.load(
                    '/braindata/abide39/train_data_' + str(fold) + '.npy')
                train_label = np.load(
                    '/braindata/abide39/train_label_' + str(fold) + '.npy')
                test_data = np.load(
                    '/braindata/abide39/test_data_' + str(fold) + '.npy')
                test_label = np.load(
                    '/braindata/abide39/test_label_' + str(fold) + '.npy')
                best_model, train_result, train_metrics = brainedgedecsttrain(model, train_data, criterion, optimizer,
                                                                            n_epochs,batch_size,train_label)
                train_results_df, train_metrics = test_train(best_model, train_data, batch_size, criterion,train_label)
                test_results_df, test_metrics = test_test(best_model, test_data,batch_size, criterion,test_label )



        print(test_results_df)
        print(test_metrics)
        train_metrics_df=pd.DataFrame.from_dict([train_metrics])
        test_metrics_df=pd.DataFrame.from_dict([test_metrics])
        FinalTestAccuracy=test_metrics_df['accuracy']*100
        FinalTestAccuracy=int(FinalTestAccuracy)

        printcsv(FinalTestAccuracy,learning_rate,n_epochs,batch_size,train_metrics_df,test_metrics_df,test_results_df,starttime)

        modelPath = '/data/models/'
        extension = ".pt"

        FinalTestAccuracy = test_metrics_df['accuracy'] * 100
        FinalTestAccuracy = int(FinalTestAccuracy)
        FinalTestAccuracy_str=str(FinalTestAccuracy)
        import time
        timestr = time.strftime("%Y%m%d-%H%M%S")
        Model_file = modelPath + FinalTestAccuracy_str + '_4abide_dedgeconv_Mixup_dec_st' + timestr + extension



        torch.save(best_model.state_dict(), Model_file)



