from transformers import pipeline

from transformers import AutoTokenizer, BertModel
import torch
import numpy as np
import json
import sys
from config import *

import os
import shutil


# ----------------------------------------------------------------------------
#
#      通过id映射生成的位置生成numpy array并存入npy
#
# ----------------------------------------------------------------------------


def main():
    print("-----------------------------------------------------------------------"
          "-----\n\n通过id映射生成的位置生成numpy array并存入npy\n\n------------"
          "----------------------------------------------------------------")
    # if os.path.exists(cfg.questions_BERT_embedding_file_name):
    #     return
    print(os.getcwd())
    if not os.path.exists(cfg.rawId2idx_json_file):
        raise RuntimeError(f"{cfg.rawId2idx_json_file}  : 需要的文件不存在")
    if not os.path.exists(cfg.all_text_questions_file):
        raise RuntimeError(f"{__file__}  : ques根目录没有需要的文件 : {cfg.all_text_questions_file}")

    with open(cfg.all_text_questions_file, "r") as jfp:
        question_info = json.load(jfp)

    # 真实id->从0开始的id
    with open(cfg.rawId2idx_json_file, "r") as ff:
        rawId2idx = json.load(ff)
    #question_info 最外层是assignment id为key，一个rawid2question的dict为value，rawid2question即表中的原始不连续quesid为key，question的文本为value
    qOfAss = {}
    for ass, id2qs in question_info.items():
        idxs = [] # per-assignment 将id2qs中的rawid转成0-9的id，根据rawId2idx这个mapping
        qs = []   # per-assignment 问题 和上面对应
        for rawid, q in id2qs.items():
            idxs.append(rawId2idx[ass][rawid])
            qs.append(q)

            # 根据位置重排，原来的重排是错的
            # e.g. 原本idx可能是 7,4,2, ....
            # 那么在ordered_qs中，第7个位置才应该填入idx列表中的第0个值对应的question
            # 也就是 ordered_qs[7] = qs[0]


        ordered_qs = ["" for i in range(10)]
        for i, idx in enumerate(idxs):
            ordered_qs[idx] = qs[i]
        qOfAss[ass] = ordered_qs

    all_ordered_qs = [0] * 5 #对应5个assignment 最终这5个位置每个位置变为一个按照0->9id排好的问题text列表
    for i, (ass, qs) in enumerate(qOfAss.items()):
        # ass_idx 0-4

        # assignment id 转换成 0-4（5个ass）的id
        # CHECK 是不是和信息表同时生成
        ass_idx = cfg.ass2idx[int(ass)]

        # 生成二维数组
        all_ordered_qs[ass_idx] = qs



    # 展平
    flatted_ordered_qs=[]
    for qss in all_ordered_qs:
        flatted_ordered_qs.extend(qss)


    tokenizer = AutoTokenizer.from_pretrained("../hf_windows/bert-base-uncased")
    print("BERT 问题 tokenizer ok")
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # model = BertModel.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("../hf_windows/bert-base-uncased")

    print("BERT model 加载 ok")
    # q用空格分隔的句子，标点也算一个word，头和尾有特殊token
    data_input = tokenizer(flatted_ordered_qs, return_tensors="pt", padding=True, truncation=True, max_length=150)
    output = model(**data_input)
    q_embeds = output.last_hidden_state[:, 0, :].detach().numpy()
    np.save(cfg.questions_BERT_embedding_file_name, q_embeds)

    print("預訓練問題向量生成完畢ok")
    # with open(final_json_file_name, "w") as json_file:
    #     json.dump(d, json_file)


if __name__ == "__main__":
    main()
