from transformers import pipeline
import shutil

from transformers import AutoTokenizer, BertModel
import torch
import numpy as np
import json
import sys
from config import *

import os


def main():
    if os.path.exists(cfg.questions_BERT_embedding_file_name):
        return

    if not os.path.exists(cfg.question_id_map_fname):
        raise RuntimeError(f"{cfg.question_id_map_fname}  : 需要的文件不存在")
    if not os.path.exists(cfg.question_file_name):
        root_file_name = os.path.join(cfg.all_question_dir, os.path.basename(cfg.question_file_name))
        print(f"从question root目录复制文件 : {root_file_name}")
        if os.path.exists(root_file_name):
            shutil.copy(root_file_name, cfg.question_file_name)
        # TODO 把旧文件复制到需要的新目录
        else:
            raise RuntimeError(f"{__file__}  : ques根目录没有需要的文件 : {root_file_name}")
    with open(cfg.question_file_name, "r") as jfp:
        question_info = json.load(jfp)

    with open(cfg.question_id_map_fname, "r") as ff:
        id_map = json.load(ff)

    ids = []
    qs = []
    for k, q in question_info.items():
        ids.append(id_map[k])
        qs.append(q)

    # reorder questions by id

    questions = [qs[i] for i in ids]
    del qs

    # feed questions into BERT

    tokenizer = AutoTokenizer.from_pretrained("../hf_windows/bert-base-uncased")
    print("ok")
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # model = BertModel.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("../hf_windows/bert-base-uncased")

    print("ok2")
    # q用空格分隔的句子，标点也算一个word，头和尾有特殊token
    data_input = tokenizer(questions, return_tensors="pt", padding=True, truncation=True, max_length=150)
    output=model(**data_input)
    q_embeds=output.last_hidden_state[:,0,:].detach().numpy()
    np.save(cfg.questions_BERT_embedding_file_name, q_embeds)

    print("ok")
    # with open(final_json_file_name, "w") as json_file:
    #     json.dump(d, json_file)


if __name__ == "__main__":
    main()
