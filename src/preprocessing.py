import math
import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from config import *
import json

from json_utils import *


def main(use_all_assignments=False):
    main_df = pd.read_csv(config.raw_main_table)
    back_df = pd.read_csv(config.raw_main_table)
    main_df = main_df[main_df["EventType"] == "Run.Program"]
    if not use_all_assignments:
        main_df = main_df[main_df["AssignmentID"] == Config.ASSIGNMENT_ID]

    # 修改id数据类型，统一为int型str
    # -------------------------------------------------------------------------
    #
    #   iDX为int，raw均为str
    #
    #
    # -------------------------------------------------------------------------

    for column in main_df.columns:
        if "AssignmentID" == column or "ProblemID" == column:
            if main_df[column].dtype == float:
                main_df[column] = main_df[column].astype(int)
                main_df[column] = main_df[column].astype(str)
    # def find_files_with_439(directory):
    #     files_with_439 = []
    #     for root, dirs, files in os.walk(directory):
    #         for file in files:
    #             if '439' in file:
    #                 files_with_439.append(os.path.join(root, file))
    #     return files_with_439
    #
    # directory = '/path/to/directory'
    # files = find_files_with_439(directory)
    # print(files)

    # cfg.assignments = [439, 487, 492, 494, 502]

    # def load_json(f):
    #     with open(f, 'r') as ff:
    #         data = json.load(ff)
    #     return data
    #
    # def save_dict_as_json(data,file):
    #     with open(file, 'w') as f:
    #         json.dump(data, f)

    all_probs = {}
    for root, dirs, files in os.walk(cfg.all_question_dir):
        for qf in files:
            for ass in cfg.assignments:
                if str(ass) in qf:
                    all_probs[ass] = load_json(os.path.join(root, qf))

    save_dict_as_json(all_probs, cfg.all_text_questions_file)
    # true_id <-> npy idx
    # ASS直接硬编码映射
    # 所以写先用raw_ass_id写，再改成ass_idx
    rawId2idx = {}
    idx2rawId = {}
    for ass_id, probs in all_probs.items():
        rawId2idx[ass_id] = {}
        idx2rawId[ass_id] = {}

        for i, pb in enumerate(probs):
            rawId2idx[ass_id][pb] = i
            idx2rawId[ass_id][i] = pb
    save_dict_as_json(rawId2idx, cfg.rawId2idx_json_file)
    save_dict_as_json(idx2rawId, cfg.idx2rawId_json_file)

    # TODO rewrite
    # add a new column to raw file
    # idx_column = []

    aaa = pd.unique(main_df["AssignmentID"])

    def generate_idx(row):
        ass_id = row["AssignmentID"]
        prob_id = row["ProblemID"]
        # pre-bug KeyError: 32
        return rawId2idx[int(ass_id)][str(prob_id)]

    # for i in range(len(main_df)):
    #     row = main_df.iloc[i]
    #     ass_id = row["AssignmentID"]
    #     prob_id = row["ProblemID"]
    #     # pre-bug KeyError: 32
    #     idx_column.append(rawId2idx[int(ass_id)][str(prob_id)])

    main_df["prob_idx"] = main_df.apply(generate_idx, axis=1)
    back_df["prob_idx"] = back_df.apply(generate_idx, axis=1)
    back_df.to_csv(config.post_preprocess_main_table)
    if False:
        problems = pd.unique(main_df["ProblemID"])

        pp = list(map(int, problems))
        problems_d = {true_id: idx for idx, true_id in enumerate(pp)}
        with open(cfg.question_id_map_fname, "w") as json_file:
            json.dump(problems_d, json_file)
        """
             TODO 本脚本对数据集做划分，并没有并入code那张表
        """

    students = pd.unique(main_df["SubjectID"])

    d = {}
    for s in students:
        d[s] = {}
        # main_df已经筛出了assignment439，理论上这个文件生成的都肯定是439里的
        df = main_df[main_df["SubjectID"] == s]

        # ***************************************************
        #
        #                         加入排序
        #
        # ***************************************************
        df=df.sort_values(by='ServerTimestamp')
        d[s]["length"] = len(df)
        # d[s]["Problems"] = [str(problems_d[i]) for i in df["ProblemID"]]

        # 写入question的idx
        # why str?
        # TypeError: sequence item 0: expected str instance, int found
        d[s]["Problems"] = list(df["prob_idx"].astype(str))
        d[s]["Result"] = list((df["Score"] == 1).astype(int).astype(str))
        # d[s]["CodeStates"] = list(df["CodeStateID"])
        # me

        # update add assignment_id
        d[s]["AssignmentID"] = list(df["AssignmentID"])
        d[s]["CodeStates"] = [str(item) for item in df["CodeStateID"]]

        # s : 學生id(勘误，是序列长度)
        # CodeStates' 一个列表，每个element是code id对应另一张表里的code
        # problems   一个题号列表？元素有重复，重复代表一个problem的多次尝试
        #
        # result   一个列表，代表每次尝试正确与否
        # length    尝试次数

    train_val_s, test_s = train_test_split(students, test_size=0.2, random_state=1)
    # TODO 随机划分在这里，将stuid划分了一下
    # processed_data_dir
    # np.save("../data/training_students.npy", train_val_s)
    # np.save("../data/testing_students.npy", test_s)
    np.save(os.path.join(cfg.processed_data_dir, "training_students.npy"), train_val_s)
    np.save(os.path.join(cfg.processed_data_dir, "testing_students.npy"), test_s)
    # if not os.path.isdir("../data/DKTFeatures"):
    #     os.mkdir("../data/DKTFeatures")

    file_test = open(os.path.join(cfg.splited_data_dir, "test_data.csv"), "w")
    for s in test_s:
        if d[s]['length'] > 0:
            file_test.write(str(d[s]['length']))
            file_test.write(",\n")
            file_test.write(",".join(d[s]['CodeStates']))
            file_test.write(",\n")
            file_test.write(",".join(d[s]['Problems']))
            file_test.write(",\n")
            file_test.write(",".join(d[s]['Result']))
            file_test.write(",\n")
            # add
            file_test.write(",".join(d[s]['AssignmentID']))
            file_test.write(",\n")

    for fold in range(cfg.fold):
        train_s, val_s = train_test_split(train_val_s, test_size=0.25, random_state=fold)

        file_train = open(os.path.join(cfg.splited_data_dir, "train_firstatt_" + str(fold) + ".csv"), "w")
        for s in train_s:
            if d[s]['length'] > 0:
                file_train.write(str(d[s]['length']))
                file_train.write(",\n")
                file_train.write(",".join(d[s]['CodeStates']))
                file_train.write(",\n")
                file_train.write(",".join(d[s]['Problems']))
                file_train.write(",\n")
                file_train.write(",".join(d[s]['Result']))
                file_train.write(",\n")
                # add
                file_train.write(",".join(d[s]['AssignmentID']))
                file_train.write(",\n")
        file_val = open(os.path.join(cfg.splited_data_dir, "val_firstatt_" + str(fold) + ".csv"), "w")
        for s in val_s:
            if d[s]['length'] > 0:
                file_val.write(str(d[s]['length']))
                file_val.write(",\n")
                file_val.write(",".join(d[s]['CodeStates']))
                file_val.write(",\n")
                file_val.write(",".join(d[s]['Problems']))
                file_val.write(",\n")
                file_val.write(",".join(d[s]['Result']))
                file_val.write(",\n")
                # add
                file_val.write(",".join(d[s]['AssignmentID']))
                file_val.write(",\n")
    print("preprocessing DONE")


if __name__ == "__main__":
    main()
