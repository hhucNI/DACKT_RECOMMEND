import os
import random

import torch

import torch.optim as optim
import numpy as np

from dataloaderOriginal import get_data_loader
import evaluationSimpleChange as evaluation
import warnings

from LSTMoriginal import c2vRNNModel
from config import *

warnings.filterwarnings("ignore")

import train_utils


def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    config = Config()
    print("------------------------------------------------------------------------------")
    print("-------------------------          LSTM MODEL    ---------------------------")
    print("------------------------------------------------------------------------------")
    setup_seed(0)

    if torch.cuda.is_available():
        print("$$$$$$$$$$$$$$$$$$$$$   CUDA 可用")
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model_type = "lstm"
    model_type = model_type + f"[{L_EPOCH}]"
    print(f"model type : {model_type}")
    performance_list = []
    scores_list = []
    first_scores_list = []
    first_total_scores_list = []
    saved_results = evaluation.init_json_dict(config)

    for fold in range(10):
        print("----", fold, "-th run----")
        train_loader, test_loader = get_data_loader(config.bs, config.questions, config.length, fold)
        node_count, path_count = np.load(os.path.join(config.processed_data_dir, "np_counts.npy"))

        model = c2vRNNModel(config.questions * 2,
                            config.hidden,
                            config.layers,
                            config.questions,
                            node_count, path_count, device)

        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        loss_func = evaluation.lossFunc(config.questions, config.length, device)
        all_epoch_loss = []

        for epoch in range(lstm_config.lstm_epochs):
            print('epoch: ' + str(epoch))
            model, optimizer, epoch_loss = evaluation.train_epoch(model, train_loader, optimizer,
                                                                  loss_func, config, device, model_type=model_type)
            all_epoch_loss.append(epoch_loss)

        train_utils.save_losses_per_epoch(all_epoch_loss, model_type=model_type)

        first_total_scores, first_scores, scores, performance = evaluation.test_epoch(
            model, test_loader, loss_func, device, epoch, config, fold, model_type)
        first_scores = {k: list(v) for k, v in first_scores.items()}
        scores = {k: list(v) for k, v in scores.items()}
        evaluation.save_result_per_fold(fold, saved_results, list(performance), list(first_total_scores),
                                        first_score=first_scores, score=scores)

        first_total_scores_list.append(first_total_scores)
        scores_list.append(scores)
        first_scores_list.append(first_scores)
        performance_list.append(performance)
    first_avg = np.mean(first_total_scores_list, axis=0)
    print("Average scores of the first attempts:", first_avg)
    all_avg = np.mean(performance_list, axis=0)
    print("Average scores of all attempts:", all_avg)
    evaluation.save_result_json(saved_results, config, list(first_avg), list(all_avg), model_type=model_type)


if __name__ == '__main__':
    main()
