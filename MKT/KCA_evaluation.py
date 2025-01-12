import os.path

import MyUtils
import tqdm
import torch
import logging
import seaborn as sns
import numpy as np
import matplotlib.pylab as plt
import torch.nn as nn
import time
from sklearn import metrics
import os

from eval_utils import *




def train_epoch_LSTM(model, trainLoader, optimizer, loss_func, config, device,model_type):
    model.to(device)
    epoch_loss=[]
    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        batch_new = batch[:, :-1, :].to(device)
        pred = model(batch_new)
        loss, prediction, ground_truth = loss_func(pred, batch[:, :, :config.questions * 2])
        optimizer.zero_grad()
        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    return model, optimizer,sum(epoch_loss)/len(epoch_loss)


def test_epoch_LSTM(model, testLoader, loss_func, device, epoch, config, fold,model_type):
    print(f"fold {fold} one test loader :    ========   Fold   ==========")

    model.to(device)
    ground_truth = torch.tensor([])
    prediction = torch.tensor([])
    full_data = torch.tensor([])
    preds = torch.tensor([])
    batch_n = 0
    for batch in tqdm.tqdm(testLoader, desc='Testing:     ', mininterval=2):
        batch_new = batch[:, :-1, :].to(device) # 输入pred 49维[1:50]位置的预测  batch 50维
        pred = model(batch_new) # pred (64,49,10) # 这边是每个step所有问题上的预测都有
        total_size=pred.shape[0]*pred.shape[1]
        # 但实际上每个step只有一个问题 #pred 49 [1:50] batch 50 [0:50]
        loss, p, a = loss_func(pred, batch) # p (2144,) <=64*49

        prediction = torch.cat([prediction, p]) # 这个是展平了的一维的，学生做题序列题号对应的prediction，
        ground_truth = torch.cat([ground_truth, a]) # 这个是展平了的一维的，学生做题序列题号对应的ground truth，
        full_data = torch.cat([full_data, batch])
        preds = torch.cat([preds, pred.cpu()]) # 没有展平，每个题号位置都有预测，模型原始输出的拼接 大致是(78,49,10)
        # plot_heatmap(batch, pred, fold, batch_n)
        batch_n += 1
        pred_check(p,model_type,fold,batch_n,total_size,actual_size=p.shape[0]) # 自定义函数

    return performance_granular(full_data, preds, ground_truth, prediction, fold, config)







def performance_granular(batch, pred, ground_truth, prediction, fold, config):
    """
     评估一个test文件的性能
     first_ : 第一次尝试的各项指标，每个问题对应一个（那个循环
     first_total_ 总的第一次尝试的各项指标，上面是一维，这个total就是二维，理解为取个平均
     普通，就是所有尝试的指标，每个问题对应一个
     参数 大概相当于普通的平均
     score : 所有指标的一个list [auc,f1,recall,precision,acc]
    """
    # if not os.path.exists("logs"):
    #     os.mkdir("logs")
    # log_file_name=f"logs/bos_transformer_model_eval_{fold}fold.log"
    # if os.path.exists(log_file_name):
    #     os.remove(log_file_name)
    # logfile=open(log_file_name,"a")
    # logfile.write("\n\n\n -----------------BOS--transformer---------------------------------------------\n")
    # preds，问题id为key，所有人在这个问题上尝试的pred list为value
    preds = {k:[] for k in range(config.questions)}

    # ground truth 问题id为key，所有人在这个问题上的ground truth list(1 or 0)
    gts = {k:[] for k in range(config.questions)}
    first_preds = {k:[] for k in range(config.questions)}
    first_gts = {k:[] for k in range(config.questions)}
    scores = {}
    first_scores = {}


    for s in range(pred.shape[0]):
        # pred (78batch_num,49,10)
        # 这里的s表示一个学生，i.e. batch[s]表示一个student的答题序列

        # 关于这里的错位问题
        # pred是每个时间步网络输出，取前49个，0-1中的一个浮点数

        # a是实际上答对了与否，也就是0 or 1，也就是ground truth
        # 因为问题是预测下一个，所以为了与pred对应，去掉第一个，取[1:50]个

        # delta是每个time step答的是哪题，把表示答对的10个位置和表示答错的10个位置堆起来
        # 为了获取每次输出在 正确位置(time step) 的预测结果

        # pred[student][:,i]每一列代表在第i个问题上的预测结果
        # 所以delta要与a保持一致，取[1:50]
        # delta (49,10)
        delta = (batch[s][1:, 0:config.questions] +
                 batch[s][1:, config.questions:config.questions*2])
        #具体参见 平板->nebo笔记本->研究方向相关->idea->codedkt代码解释，评估部分


        # temp (49,49)
        # temp每列是一个问题在所有时间步上的预测(列向量维度49)，至于是哪个问题则看答题序列
        # temp[i][j] : 第i个时间步时，在j这个问题上的pred，至于j是哪个问题则看答题序列
        temp = pred[s][:config.length-1].mm(delta.T)

        index = torch.tensor([[i for i in range(config.length-1)]],
                             dtype=torch.long)
        # 这一步是取temp的对角线 也就是每个时间步在作答的那题上的预测值
        p = temp.gather(0, index)[0].detach().cpu().numpy()
        a = (((batch[s][:, 0:config.questions] - batch[s][:, config.questions:config.questions*2]).sum(1) + 1) // 2)[1:].detach().cpu().numpy()

        for i in range(len(p)):
            if p[i] > 0:
                p = p[i:]
                a = a[i:]
                delta = delta.detach().cpu().numpy()[i:]
                break

        # 这两层循环是处理per problem评估
        # delta (49,10)
        for i in range(len(p)):
            for j in range(config.questions):
                if delta[i,j] == 1:
                    # 当确定是j这道题时
                    # 由于p,a已经是对应题目上的预测序列(序列基于时间步)
                    # 直接加入p[i]和a[i],i代表时间步

                    # 注意preds[j]是一个列表，把p[i] append进去
                    # preds的每个位置代表每个问题下的 pred
                    preds[j].append(p[i])
                    gts[j].append(a[i])
                    if i == 0 or delta[i-1,j] != 1:
                        # 这里条件比上面进一步收紧，还要保证是序列第一个或者是换题的时刻
                        # 上一题答的不是本题，也就是比如上一题是question2，这次变question3了
                        first_preds[j].append(p[i])
                        first_gts[j].append(a[i])

    first_total_gts = []
    first_total_preds = []

    for j in range(config.questions):
        info0=f"\t Problem {j+1}  : "
        print(info0)
        # logfile.write(info0+"\n")
        # 在每个问题下，预测的各项指标，并不是按照时间顺序或者人物划分，只看问题
        # e.g. 所有学生在一个问题上的尝试，被融合到一起来计算指标
        f1 = metrics.f1_score(gts[j], np.where(np.array(preds[j])>0.5, 1, 0))
        recall = metrics.recall_score(gts[j], np.where(np.array(preds[j])>0.5, 1, 0))
        precision = metrics.precision_score(gts[j], np.where(np.array(preds[j])>0.5, 1, 0))
        acc = metrics.accuracy_score(gts[j], np.where(np.array(preds[j])>0.5, 1, 0))
        try:
            auc = metrics.roc_auc_score(gts[j], preds[j])
        except ValueError:
            auc = 0.5
        # 一个返回值 score ,所以可以看出score是双层的
        # 第一层是问题
        # 第二层是 [auc,f1,recall,precision,acc]
        scores[j]=[auc,f1,recall,precision,acc]

        # loginfo='problem '+str(j)+'\n auc: ' + str(auc) + '\n f1: ' + str(f1) + '\n recall: ' + str(recall) +'\n precision: ' + str(precision) + '\n acc: ' +str(acc)
        loginfo='\t\t not first attempt : problem level'+str(j)+' auc: ' + str(auc) + ' f1: ' + str(f1) + ' recall: ' + str(recall) +' precision: ' + str(precision) + ' acc: ' +str(acc)
        print(loginfo)
        # logfile.write(loginfo+"\n")

        # logfile.write("time : "+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+"\n")
        # logfile.write(loginfo+"\n\n")
        # print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        first_f1 = metrics.f1_score(first_gts[j], np.where(np.array(first_preds[j])>0.5, 1, 0))
        first_recall = metrics.recall_score(first_gts[j], np.where(np.array(first_preds[j])>0.5, 1, 0))
        first_precision = metrics.precision_score(first_gts[j], np.where(np.array(first_preds[j])>0.5, 1, 0))
        first_acc = metrics.accuracy_score(first_gts[j], np.where(np.array(first_preds[j])>0.5, 1, 0))
        try:
            first_auc = metrics.roc_auc_score(first_gts[j], first_preds[j])
        except ValueError:
            first_auc = 0.5

        # 上一步双层循环收集的per problem的first_gts 全塞进一个total里，
        # 相当于做个平均  (extend)
        first_total_gts.extend(first_gts[j])
        first_total_preds.extend(first_preds[j])


        # 一个返回值 first_score ,所以可以看出first_score是双层的
        # 第一层是问题
        # 第二层是 [first_auc,first_f1,first_recall,first_precision,first_acc]
        first_scores[j]=[first_auc,first_f1,first_recall,first_precision,first_acc]
        info2='\t\t First prediction for problem '+str(j)+' auc: ' + str(first_auc) + ' f1: ' + str(first_f1) + ' re'
        'call: ' + str(first_recall) + ' precision: ' + str(first_precision) + ' acc: ' + str(first_acc)
        print(info2)
        print("")
        # logfile.write(info2+"\n")


    #这些就是最基本的模型输出计算出的结果，这里的prediction已经是相应位置的预测了
    # 不需要任何处理即可直接计算，注意到上面的所有代码都与下面这几行无关
    f1 = metrics.f1_score(ground_truth.detach().numpy(),
                          torch.where(prediction>0.5,torch.tensor(1), torch.tensor(0)).detach().numpy())
    recall = metrics.recall_score(ground_truth.detach().numpy(),
                                  torch.where(prediction>0.5,torch.tensor(1), torch.tensor(0)).detach().numpy())
    precision = metrics.precision_score(
        ground_truth.detach().numpy(),
        torch.where(prediction>0.5,torch.tensor(1), torch.tensor(0)).detach().numpy())
    acc = metrics.accuracy_score(
        ground_truth.detach().numpy(),
        torch.where(prediction>0.5,torch.tensor(1), torch.tensor(0)).detach().numpy())
    auc = metrics.roc_auc_score(
        ground_truth.detach().numpy(),
        prediction.detach().numpy())
    # loginfo_common='\nauc: ' + str(auc) + '\n f1: ' + str(f1) + '\n recall: ' + str(recall) + '\n precision: ' + str(precision) + '\n acc: ' +str(acc)
    info3="\n\n \t============   Total  :   ==========\n"
    print(info3)
    # logfile.write(info3+"\n")
    # loginfo_common='\nauc: ' + str(auc) + '\n f1: ' + str(f1) + '\n recall: ' + str(recall) + '\n precision: ' + str(precision) + '\n acc: ' +str(acc)
    loginfo_common='auc: ' + str(auc) + ' f1: ' + str(f1) + ' recall: ' + str(recall) + ' precision: ' + str(precision) + ' acc: ' +str(acc)
    print(loginfo_common)
    # logfile.write(loginfo_common+"\n")


    # 这些就是把per problem的全展开，放在一个一维列表里的结果
    # 相当于做个平均
    first_total_f1 = metrics.f1_score(first_total_gts, np.where(np.array(first_total_preds)>0.5, 1, 0))
    first_total_recall = metrics.recall_score(first_total_gts, np.where(np.array(first_total_preds)>0.5, 1, 0))
    first_total_precision = metrics.precision_score(first_total_gts, np.where(np.array(first_total_preds)>0.5, 1, 0))
    first_total_acc = metrics.accuracy_score(first_total_gts, np.where(np.array(first_total_preds)>0.5, 1, 0))
    try:
        first_total_auc = metrics.roc_auc_score(first_total_gts, first_total_preds)
    except ValueError:
        first_total_auc = 0.5

    first_total_scores = [first_total_auc,first_total_f1,first_total_recall,first_total_precision,first_total_acc]
    # logfile.close()
    # total ----per quesition ------ per question -----total
    return first_total_scores, first_scores, scores, [auc,f1,recall,precision,acc]

def plot_heatmap(batch, pred, fold, batch_n, config):

    # TODO: No hardcoding problem dict but what about other assignments?
    problem_dict = {"000000010":"1",
                    "000000001":"3",
                    "000010000":"5",
                    "010000000":"13",
                    "001000000":"232",
                    "000100000":"233",
                    "100000000":"234",
                    "000001000":"235",
                    "000000100":"236"
                   }
    problems = []
    for s in range(pred.shape[0]):

        delta = (batch[s][1:, 0:config.questions] + batch[s][1:, config.questions:config.questions*2]).detach().cpu().numpy()

        a = (((batch[s][:, 0:config.questions] - batch[s][:, config.questions:config.questions*2]) + 1) // 2)[1:].detach().cpu().numpy()
        p = pred[s].detach().cpu().numpy()

        for i in range(len(delta)):
            if np.sum(delta, axis=1)[i] > 0:
                p = p[i:]
                a = a[i:]
                delta = delta[i:]
                break

        problems = [problem_dict["".join([str(int(i)) for i in sk])] for sk in delta]

        plt.figure(figsize=(15, 6), dpi=80)

        ax = sns.heatmap(p.T, annot=a.T, linewidth=0.5, vmin=0, vmax=1, cmap="Blues")

        plt.xticks(np.arange(len(problems))+0.5, problems, rotation=45)
        plt.yticks(np.arange(config.questions)+0.5, ['234', '13', '232', '233', '5', '235', '236', '1', '3'], rotation=45)
        plt.xlabel("Attempting Problem")
        plt.ylabel("Problem")


        plt.title("Heatmap for student "+str(s)+" fold "+str(fold))
        plt.tight_layout()
        plt.savefig("heatmaps/b"+str(batch_n)+"_s"+str(s)+"_f"+str(fold)+".png")



class lossFunc(nn.Module):
    def __init__(self, num_of_questions, max_step, device):
        super(lossFunc, self).__init__()
        self.crossEntropy = nn.BCELoss()
        self.num_of_questions = num_of_questions
        self.max_step = max_step
        self.device = device


    def forward(self, pred, batch):
        """
         TODO pred包含了任意位置（question)上的预测结果
                    但是给定的y（即这里的a中），仅有对应位置的ground_truth
                    所以计算loss只能用到部分信息，还有这也是heatmap的形成原因
        """
        loss = 0
        prediction = torch.tensor([])
        ground_truth = torch.tensor([])
        pred = pred.to('cpu')

        for student in range(pred.shape[0]):
            #delta表示第几题做了，把正确和错误两个矩阵相加了
            # 此处pred和batch均为50？
            delta = batch[student][:, 0:self.num_of_questions] + batch[
                student][:, self.num_of_questions:self.num_of_questions*2]  # shEape: [length, questions]
            # 关于这里的错位问题
            # pred是每个时间步网络输出，取前49个
            # a是实际上答对了与否，因为问题是预测下一个，所以为了与pred对应，去掉第一个，取[1:50]个
            # delta是每个time step答的是哪题，为了获取每次输出在 正确位置(time step) 的预测结果
            # pred[student][:,i]每一列代表在第i个问题上的预测结果
            # 所以delta要与a保持一致，取[1:50]

            # 具体参见 平板->nebo笔记本->研究方向相关->idea->codedkt代码解释，评估部分

            # temp (49,49)
            # temp每列是一个问题在所有时间步上的预测(列向量维度49)，至于是哪个问题则看答题序列
            # temp[i][j] : 第i个时间步时，在j这个问题上的pred，至于j是哪个问题则看答题序列
            temp = pred[student][:self.max_step-1].mm(delta[1:].t())
            index = torch.tensor([[i for i in range(self.max_step-1)]],
                                 dtype=torch.long)
            # 取temp的对角线,p
            # a即每个学生的ground truth
            # 每个step，对next时间步所在的问题的预测结果 #batch[student] : (50,10)
            # 那么很明显，sum(1)去掉的是10这个维度，也就是把question维度去掉，再结合+1/2，
            # 原来是-1 or 1 现在是1 or 0
            p = temp.gather(0, index)[0]
            a = (((batch[student][:, 0:self.num_of_questions] -
                   batch[student][:, self.num_of_questions:self.num_of_questions*2]).sum(1) + 1) //
                 2)[1:]

            # fixme 由于数据输入过程中的padding操作是在前面padding，所以对应位置的pred均为0，直到不为0出现，说明接下来的position是有效的
            for i in range(len(p)):
                if p[i] > 0:
                    p = p[i:]
                    a = a[i:]
                    break
            # 每次循环计算一个学生的序列实际长度，并且concat，
            # 最终concat 返回的结果即 一个batch所有学生一共的 实际submission次数
            loss += self.crossEntropy(p, a) # p,a (36,) 36并非确定值，小于等于最大序列长度50
            prediction = torch.cat([prediction, p])
            ground_truth = torch.cat([ground_truth, a])

        return loss, prediction, ground_truth



def train_epoch(model, trainLoader, optimizer, loss_func, config, device,model_type):
    model.to(device)
    epoch_loss=[]
    # 该任务的y隐藏在输入x中了
    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        # TODO 查看输入数据（可能已经经过预处理)情况
        #batch (128,50,320)
        # batch_new (128,49,320)
        # 只需把前49个送入GPU
        # 注意下面的ground_truth是用原始batch算的，不会造成y的丢失


        # batch_new = batch[:,:-1,:].to(device)

        #my change   2023-9-22
        # (64,50,320)
        pred = model(batch)
        loss, prediction, ground_truth = loss_func(pred, batch[:,:,:config.questions*2])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
    return model, optimizer,sum(epoch_loss)/len(epoch_loss)

@DeprecationWarning
def train_epoch_for_transformer(model, trainLoader, optimizer, loss_func, config, device):
    model.to(device)

    # 该任务的y隐藏在输入x中了
    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        # TODO 查看输入数据（可能已经经过预处理)情况
        #batch (128,50,320)
        # batch_new (128,49,320)
        # 只需把前49个送入GPU
        # 注意下面的ground_truth是用原始batch算的，不会造成y的丢失
        batch_new = batch.to(device)
        # batch_new = batch[:,:-1,:].to(device)
        pred = model(batch_new)
        #out 推测size(128,tgt_length=49,transformer_encoding_dim=?自定义超参)
        # 需要一个linear projector

        # batch[:,:,:config.questions*2] 用于生成ground_truth
        loss, prediction, ground_truth = loss_func(pred, batch[:,:,:config.questions*2])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, optimizer

# fixme 待修改，注意各维度匹配，还有注意loss_func里的维度匹配
def test_epoch_for_transformer(model, testLoader, loss_func, device, epoch, config, fold):
    model.to(device)
    ground_truth = torch.tensor([])
    prediction = torch.tensor([])
    full_data = torch.tensor([])
    preds = torch.tensor([])
    batch_n = 0
    for batch in tqdm.tqdm(testLoader, desc='Testing:     ', mininterval=2):
        batch_new = batch.to(device)
        pred = model(batch_new)
        loss, p, a = loss_func(pred, batch)

        prediction = torch.cat([prediction, p])
        ground_truth = torch.cat([ground_truth, a])
        full_data = torch.cat([full_data, batch])
        preds = torch.cat([preds, pred.cpu()])
        # plot_heatmap(batch, pred, fold, batch_n)
        batch_n += 1

    return performance_granular(full_data, preds, ground_truth, prediction, epoch, config)


def test_epoch(model, testLoader, loss_func, device, epoch, config, fold):
    model.to(device)
    ground_truth = torch.tensor([])
    prediction = torch.tensor([])
    full_data = torch.tensor([])
    preds = torch.tensor([])
    batch_n = 0
    for batch in tqdm.tqdm(testLoader, desc='Testing:     ', mininterval=2):
        batch_new = batch[:,:-1,:].to(device)
        #my change
        pred = model(batch_new,batch)
        loss, p, a = loss_func(pred, batch)

        prediction = torch.cat([prediction, p])
        ground_truth = torch.cat([ground_truth, a])
        full_data = torch.cat([full_data, batch])
        preds = torch.cat([preds, pred.cpu()])
        # plot_heatmap(batch, pred, fold, batch_n)
        batch_n += 1
    # 评估一整个测试文件
    return performance_granular(full_data, preds, ground_truth, prediction, epoch, config)
#
# def expand_tgt(x,d_model):
#     x = x.view(x.size(0), x.size(1), 1)
#     res = x.expand(-1, -1, d_model)
#     return res
def test_epoch_for_new_transformer(model, testLoader, loss_func, device, epoch, config, fold, trans_config,model_type):
    print(f"fold {fold} one test loader :    ========   Fold   ==========")
    model.to(device)
    ground_truth = torch.tensor([])
    prediction = torch.tensor([])
    full_data = torch.tensor([])
    preds = torch.tensor([])
    batch_n = 0
    for batch in tqdm.tqdm(testLoader, desc='Testing:     ', mininterval=2):
        # batch_new = batch[:,:-1,:].to(device)
        # transformer inference one by one i.e. use a for loop 修改start token
        # TODO 训练时也要修改(输入序列加入 start token)，保持train和eval一致
        # pred_tgt = torch.randn(batch.size(0), 1).to(device)
        # pred_tgt =
        # pred_tgt_expanded = pred_tgt.view(pred_tgt.size(0), -1, 1).expand(-1, -1, trans_config.simple_d_model).to(device)
        pred_tgt_expanded = MyUtils.gen_start_token(device, batch.size(0), 1, BCQ_config.after_trans_alignment_d_model)
        # preds_one_seq=torch.tensor().to(device)
        all_probs = torch.randn(batch.size(0), 1,10).to(device)
        # (128,49,10)也要堆起来
        for i in range(config.length - 1):

            # 49(config.length - 1)+1(初始化start token)

            # 此处tgt应从0开始慢慢拼接长大
            raw_pred = model(batch, pred_tgt_expanded,evaluating=True)
            this_step = raw_pred[:, -1]  # 128,10
            all_probs=torch.cat([all_probs,this_step.view(this_step.size(0), 1, -1)],dim=1)
            pred_cur_step = torch.argmax(this_step, dim=1)
            pred_cur_step = pred_cur_step.view(pred_cur_step.size(0), 1, 1)
            pred_cur_step_expanded = pred_cur_step.expand(pred_cur_step.size(0), -1, BCQ_config.after_trans_alignment_d_model)
            pred_tgt_expanded=torch.cat([pred_tgt_expanded,pred_cur_step_expanded], dim=1)

            # preds_one_seq = torch.cat(
            #     [preds_one_seq, pred_cur_step.view(pred_cur_step.size(0), 1, pred_cur_step.size(1))], dim=1)
        loss_probs=all_probs[:,1:]
        total_size=loss_probs.shape[0]*loss_probs.shape[1]

        loss, p, a = loss_func(loss_probs, batch)
        prediction = torch.cat([prediction, p])
        ground_truth = torch.cat([ground_truth, a])
        full_data = torch.cat([full_data, batch])
        # pred (78batch_num,49,10)
        preds = torch.cat([preds, loss_probs.cpu()])
        # plot_heatmap(batch, pred, fold, batch_n)
        batch_n += 1
        pred_check(p,model_type,fold,batch_n,total_size,actual_size=p.shape[0])

    # 评估一整个测试文件
    return performance_granular(full_data, preds, ground_truth, prediction, fold, config)


