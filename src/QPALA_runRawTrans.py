import os
import random

import time
import torch

import torch.optim as optim
import numpy as np

from QPALA_dataloader import get_data_loader
import QPALA_evaluation as evaluation
import warnings
from config import *

from gpu_mem_track import MemTracker
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
warnings.filterwarnings("ignore")

from QPALA_OutRawTransformerModel import KC_Attention_LSTM
import train_utils


def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    print("------------------------------------------------------------------------------")
    print("----------------Ques Path Attn KT------   BERT Compressed Question TRANSFORMER MODEL    ---------------------------")
    print("------------------------------------------------------------------------------")
    print(__name__)
    model_type = "QPALA"
    model_type_subfix = f"[{QPALA_EPOCH}_{BCQL_LAYER}]"
    model_type = model_type + model_type_subfix
    print(f"model type : {model_type}")


    simple_config = BCQ_config
    print(simple_config)

    setup_seed(0)

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    gpu_tracker = MemTracker()

    performance_list = []
    scores_list = []
    first_scores_list = []
    first_total_scores_list = []

    # Mine
    saved_results = evaluation.init_json_dict(config)

    max_performance_acc = 0
    for fold in range(10):
        print("----", fold, "-th run----")
        # fold决定使用哪个train_attr_<fold>文件
        # config.bs=128(batch_size), config.questions=10, config.length=50
        train_loader, test_loader = get_data_loader(config.bs, config.questions, config.length, fold)
        node_count, path_count = np.load(os.path.join(config.processed_data_dir, "np_counts.npy"))

        # config.questions=10 * 2,
        # config.hidden=128,
        # config.layers=1,
        # config.questions=10,
        # node_count, path_count, device)
        # if config.model_type=="lstm":
        # TODO 我改的transformer
        # model = c2vTransformerModel(BCQ_config.after_trans_alignment_d_model,
        #                             config.questions * 2,
        #                             config.hidden,
        #                             config.layers,
        #                             config.questions,
        #                             node_count, path_count, device)
        model = KC_Attention_LSTM(BCQ_config.before_trans_alignment_d,
                                  config.questions * 2,
                                  config.hidden,
                                  config.layers,
                                  config.questions,
                                  node_count, path_count, device)
        print("end create model")
        optimizer = optim.Adam(model.parameters(), lr=QPALA_config.lr)
        loss_func = evaluation.lossFunc(config.questions, config.length, device)
        all_epoch_loss = []
        for epoch in range(QPALA_EPOCH):
            print('epoch: ' + str(epoch))
            # if config.model_type=="tran":
            #     model, optimizer = evaluation.train_epoch_for_transformer(model, train_loader, optimizer,
            #                                                               loss_func, config, device)

            # gpu_tracker.track()
            """
            改的transformer的trainer
            """
            model, optimizer, epoch_loss = evaluation.train_epoch_LSTM(model, train_loader, optimizer,
                                                                  loss_func, config, device, model_type=model_type)
            # model, optimizer = evaluationOriginal.train_epoch(model, train_loader, optimizer,
            #                                       loss_func, config, device)
            # gpu_tracker.track()
            all_epoch_loss.append(epoch_loss)
        # first_total_scores, first_scores, scores, performance = evaluation.test_epoch(
        # model, test_loader, loss_func, device, epoch, config, fold)

        train_utils.save_losses_per_epoch(all_epoch_loss, model_type=model_type)
        print(f"torch.cuda.memory_allocated() : : {gpu_tracker.get_allocate_usage()}")
        time.sleep(10)
        with torch.no_grad():

            # 以下四个返回值first_total_scores, first_scores......每个均由如下格式组成
            # [auc,f1,recall,precision,acc]
            first_total_scores, first_scores, scores, performance=evaluation.test_epoch_LSTM(model, test_loader, loss_func, device, epoch, config, fold, model_type)
            # first_total_scores, first_scores, scores, performance = evaluation.test_epoch_for_new_transformer(
            #     model, test_loader, loss_func, device, epoch, config, fold, simple_config, model_type)
            evaluation.save_result_per_fold(fold, saved_results, list(performance), list(first_total_scores),
                                            first_score=first_scores, score=scores)
        # gpu_tracker.track()

        # performance_acc最大时保存model
        # if performance[-1] > max_performance_acc:
        #     max_performance_acc = performance[-1]
        #     # save model
        #     cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        #     file_name = f"{cur_time}_" \
        #                 f"{max_performance_acc}_" \
        #                 f"epoch{config.epochs}_" \
        #                 f"enc{config.num_encoder_layers}_" \
        #                 f"dec{config.num_decoder_layers}_" \
        #                 f"performance_acc_ADDPE.pt"
        #     save_path = os.path.join(config.save_path, file_name)
        #     torch.save(model.state_dict(), save_path)
        #     print("save a MODEL %%%%% $$$$$$$$$$$$$$$$$$")

        # 注意 first_total_scores是一个fold
        #  first_total_scores_list是所有fold
        first_total_scores_list.append(first_total_scores)
        scores_list.append(scores)
        first_scores_list.append(first_scores)
        performance_list.append(performance)
    first_avg = np.mean(first_total_scores_list, axis=0)
    print("Average scores of the first attempts:", first_avg)
    all_avg = np.mean(performance_list, axis=0)
    print("Average scores of all attempts:", all_avg)
    evaluation.save_result_json(saved_results, config, list(first_avg), list(all_avg), model_type=model_type)

    print("-----------------------------------------------------------------")
    print(f"max_acc {max_performance_acc}")
    print("-----------------------------------------------------------------")


if __name__ == '__main__':
    main()
