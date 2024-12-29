import matplotlib.pyplot as plt
import numpy as np
from config import *
import os
import re
import json
from typing import List
import pandas as pd
import re

# "QuesTrans"  "lstm"  "RawTrans" "BERT_QUES_Trans"
ALL_MODELS = ["QuesTrans", "lstm", "RawTrans", "BERT_QUES_Trans", "BQMOD", "BCQ", "CT5Q", "CT5QNT", "CT5QER", "CT5LSTM",
              "CT5LSTMA", "BCQLSTMA", "BCQ_CT", "BCQ_LSTM",
              "BLAE", "BLAA", "KCA", "KCLSTMA", "QPKT", "QP2KT", "QPALA"]

COLOR_MAP = {"lstm": "#DC143C", "BCQLSTMA": "#7FFF00", "BCQ_LSTM": "#1E90FF", "BCQ": "yellow", "BLAE": "magenta",
             "BLAA": "#D2691E"}


# lstm 红 BCQLSTMA 绿 BCQ_LSTM蓝

# # 使用正则表达式提取第一个满足条件的子串
# pattern = r"apple\d+\)"  # 匹配以 apple 开头，后面跟着任意个数字并以右括号结尾的字符串
# match = re.search(pattern, text)


def highlight_max(s):
    """
    高亮显示每列中的最大值。
    """
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]


import os
import re

def score_tables_for_all_ass():
    score_list = ["auc", "f1", "recall", "precision", "acc"]
    for score_idx, score in enumerate(score_list):
        table_result_for_all_assignments(score,score_idx)
def table_result_for_all_assignments(score,score_idx, directory_path=None):

    if directory_path is None:
        directory_path = config.structured_data_root

        # TODO 目录要改
    output_file = os.path.join(config.all_ass_table_compare_dir, f"model_cmp_{score}.csv")
    # 定义要匹配的正则表达式
    results = {
        "Model Name": [],
    }
    num3 = r"\d{3}"
    pattern_str = f"{config.after_training_dir_prefix}{num3}"
    pattern = re.compile(pattern_str)

    def get_remaining_prefix(a, b):
        # 检查a是否是b的前缀
        if b.startswith(a):
            # 去掉b的前缀a
            remaining = b[len(a):]
            # 如果剩余部分长度大于等于3，取前三个字符
            if len(remaining) >= 3:
                return remaining[:3]
                # 如果剩余部分不足三个字符，返回全部剩余部分
            else:
                return remaining
        else:
            # 如果a不是b的前缀，返回空字符串或者错误信息
            return ""  # 或者 "a is not a prefix of b"

    # 遍历目录下的所有文件和子文件夹
    # model_set=set()
    for root, dirs, files in os.walk(config.structured_data_root):
        for dir_name in dirs:
            # match = re.match(pattern, dir_name)
            num33 = get_remaining_prefix(config.after_training_dir_prefix, dir_name)
            if num33 != "":
                matched_number = str(num33)  # 获取\d{3}所匹配的三个数字部分
                json_files = [f for f in os.listdir(os.path.join(root, dir_name)) if f.endswith('.json')]
                for filename in json_files:
                    model_name = filename.split('_')[2]

                    # 读取文件内容
                    file_path = os.path.join(directory_path, dir_name, filename)
                    with open(file_path, 'r') as file:
                        data = json.load(file)

                        # 提取first_avg和all_avg字段
                        first_avg = data["first_avg"]  # 假设first_avg在第一折中的第一个问题
                        all_avg = data['all_avg']  # 假设all_avg在第一折中的第一个问题
                        assignment = data['assignment']  # 假设all_avg在第一折中的第一个问题

                        # 存储结果
                        results["Assignment"] = assignment

                        if model_name not in results:
                            results[model_name] = {}
                        model_results = results[model_name]

                        all_ass_key = "A" + matched_number
                        if matched_number not in model_results:
                            model_results[all_ass_key] = []
                        model_results[all_ass_key].append(all_avg[score_idx])

                        first_ass_key = "F" + matched_number
                        if matched_number not in model_results:
                            model_results[first_ass_key] = []
                        model_results[first_ass_key].append(first_avg[score_idx])

                    # 创建DataFrame
    # model_set=list(model_set)
    base = None

    model_set = []
    for model in results.keys():
        v_type = str(type(results[model]))

        if "dict" in v_type:
            # 是模型
            base = results[model]
            model_set.append(model)
            break

    for model in results.keys():
        v_type = str(type(results[model]))

        if "dict" in v_type:
            if model == model_set[0]:
                continue
            # add to model list
            model_set.append(model)

            # merge to base
            for score in results[model]:
                base[score].extend(results[model][score])

    # md1=results[model_set[0]]
    # md2=results[model_set[1]]
    # for score in md1:
    #     md2[score].append(md1[score][0])

    df = pd.DataFrame(base)
    lstm_idx = 0
    for i, mn in enumerate(model_set):
        if "lstm" in mn or "LSTM" in mn:
            lstm_idx = i
    df_max_idx = df.shape[0] - 1
    if lstm_idx != df_max_idx:
        diff_row = df.iloc[df_max_idx, :] - df.iloc[lstm_idx, :]
    else:
        diff_row = df.iloc[0, :] - df.iloc[lstm_idx, :]

    # df.append(diff_row)
    df.loc['diff'] = diff_row
    model_set.append("DIFF")
    df["MODEL_name"] = model_set
    sorted_columns = sorted(df.columns)

    df_sorted = df.reindex(columns=sorted_columns)
    df_sorted.to_csv(output_file)


def get_json_info_fromNfiles(model_list):
    files = os.listdir(config.json_dir)
    # "QuesTrans"  "lstm"  "RawTrans"
    # model2file={k:None for k in model_list}
    model2file = {}
    for file in files:
        for m in model_list:
            # change 为了不同参数同一模型比较，加入epoch & layer_num(待定)
            pattern = "(?i)" + m + r"\[\d+_?\d*\]"

            if m in file or m.lower() in file:
                match = re.search(pattern, file)
                if match is not None:
                    model2file[match.group()] = file

    # 需要控制顺序or返回kv对
    ret = {}
    for model_name, model_res_file in model2file.items():
        full_file_name = os.path.join(config.json_dir, model_res_file)
        if os.path.exists(full_file_name):
            with open(full_file_name, "r") as jfp:
                ret[model_name] = json.load(jfp)

    return ret


def get_json_info_from2files(trans_json_namef=None, lstm_json_name=None):
    files = os.listdir(config.json_dir)
    # "QuesTrans"  "lstm"  "RawTrans"

    for file in files:
        if trans_json_name is None and "aw" in file:
            trans_json_name = file
        elif lstm_json_name is None and "lstm" in file:
            lstm_json_name = file

    # 文件夹下没有对应数据文件
    if lstm_json_name is None or trans_json_name is None:
        # 直接退出
        print("ERROR NO json file")
        print("EXIT")
        raise RuntimeError

    trans_json_name = os.path.join(config.json_dir, trans_json_name)
    lstm_json_name = os.path.join(config.json_dir, lstm_json_name)

    # 将json数据load进入内存
    if os.path.exists(trans_json_name):
        with open(trans_json_name, "r") as jfp:
            trans_info = json.load(jfp)

    if os.path.exists(lstm_json_name):
        with open(lstm_json_name, "r") as jfp:
            lstm_info = json.load(jfp)
    return trans_info, lstm_info


# plt.figure(figsize=(66,6))
def compare2model_per_fold(fold: int, trans: dict, lstm: dict):
    """
    条形图,先横向比对，比所有指标
    """
    trans_res = trans["all_results"][fold]["result"]
    lstm_res = lstm["all_results"][fold]["result"]
    cols = ["auc", "f1", "recall", "precision", "acc", "f_auc", "f_f1", "f_recall", "f_precision", "f_acc"]
    figs, axes = plt.subplots()

    x = np.arange(len(cols))  # the label locations
    width = 0.35  # the width of the bars

    # x - width / 2 : bar 的中点  width : bar 的宽度
    rects1 = axes.bar(x - width / 2, trans_res, width, label='trans_res')
    rects2 = axes.bar(x + width / 2, lstm_res, width, label='lstm_res')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    axes.set_ylabel('Scores')
    axes.set_title(f'general comparison of fold {fold}')
    axes.set_xticks(x)
    axes.set_xticklabels(cols)
    precision = 0.05
    axes.set_yticks(np.arange(0, 1 + precision, precision))
    axes.legend()
    fname = os.path.join(config.comparison_per_fold, f"fold{fold}_pre_total.png")
    figs.savefig(fname)
    # plt.show()


def get_all_metrics_on_one_assignment(model_results: dict):
    first_model_results = {k: v["first_avg"] for k, v in model_results.items()}
    all_model_results = {k: v["all_avg"] for k, v in model_results.items()}

    all_columns = ["auc", "f1", "recall", "precision", "acc"]
    first_columns = ["f_" + v for v in all_columns]
    for i, metric in enumerate(all_columns):
        metric_idx = i
        df = pd.DataFrame(columns=["Overall", "First Attempts"])
        for (m_name, all_data), (m_name, first_data) in zip(all_model_results.items(), first_model_results.items()):
            filled_in = [all_data[metric_idx], first_data[metric_idx]]
            filled_in = [f"{v * 100:.2f}%" for v in filled_in]
            df.loc[m_name] = filled_in
        fname = os.path.join(config.comparison_avg_no_del, f"{Config.ASSIGNMENT_ID}{metric}_Table.csv")

        df.to_csv(fname, index=True)


def compareNmodel_avg_total(model_results: dict, key: str):
    cols = ["auc", "f1", "recall", "precision", "acc"]
    prefix = "f_" if "f" in key else ""
    cols = [prefix + s for s in cols]
    figs, ax = plt.subplots()

    x = np.arange(len(cols))  # the label locations

    model_num = len(model_results)
    width = 0.15
    bars = []
    for i, (model_name, info) in enumerate(model_results.items()):
        Y = info[key]
        color = None
        mn = model_name.split("[")[0]
        if mn in COLOR_MAP:
            color = COLOR_MAP[mn]
        # 中心点位置
        bars.append(
            ax.bar(x - (model_num * width) / 2 + ((2 * i + 1) * width / 2), Y, width, label=model_name, color=color))

    flatted_bars = []
    for i in range(len(x)):
        for bar in bars:
            flatted_bars.append(bar[i])

    model_num = len(model_results)
    model_height_history = [3] * model_num

    for i, bar in enumerate(flatted_bars):
        cur_height = bar.get_height()
        print(cur_height)
        # 检查当前柱状条的高度和相邻柱状条的高度差
        offset = 3
        if i % model_num != 0:  # 不是一组的第一个
            if abs(cur_height - flatted_bars[i - 1].get_height()) < 0.08:  # 如果高度差较小，增加偏移量
                offset = model_height_history[(i - 1) % model_num] + 8
        model_height_history[i % model_num] = offset

        ax.annotate(f'{cur_height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, cur_height),
                    xytext=(0, offset),  # 动态调整偏移量
                    textcoords="offset points",
                    ha='center', va='bottom')
        if i % model_num == model_num - 1:
            # refresh model height  history
            model_height_history = [3] * model_num
    ax.set_ylabel('scores')
    ax.set_title(f'AVG comparison {key}')
    ax.set_xticks(x)
    ax.set_xticklabels(cols)
    precision = 0.05
    ax.set_yticks(np.arange(0, 1 + precision, precision))
    ax.legend()
    fname = os.path.join(config.comparison_avg_no_del, f"{key}.png")
    figs.savefig(fname)


def compare2model_avg_total(trans: dict, lstm: dict, key: str):
    # trans_avg=trans["first_avg"]
    lstm_res = lstm[key]
    trans_res = trans[key]
    prefix = "f_" if "f" in key else ""

    cols = ["auc", "f1", "recall", "precision", "acc"]
    cols = [prefix + s for s in cols]

    figs, ax = plt.subplots()

    x = np.arange(len(cols))  # the label locations
    width = 0.35  # the width of the bars

    rects1 = ax.bar(x - width / 2, trans_res, width, label='trans_res')
    rects2 = ax.bar(x + width / 2, lstm_res, width, label='lstm_res')
    bars = []
    for r1, r2 in zip(rects1, rects2):
        bars.append(r1)
        bars.append(r2)
    for bar in bars:
        height = bar.get_height()
        print(height)
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset for better readability
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('scores')
    ax.set_title(f'AVG comparison {key}')
    ax.set_xticks(x)
    ax.set_xticklabels(cols)
    precision = 0.05
    ax.set_yticks(np.arange(0, 1 + precision, precision))
    ax.legend()
    fname = os.path.join(config.comparison_avg_no_del, f"{key}.png")
    figs.savefig(fname)


# for fold in range(config.fold):
# show_fold_total(fold,trans_info,lstm_info)

# compare_avg_total(trans_info,lstm_info,"first_avg")
# compare_avg_total(trans_info,lstm_info,"all_avg")

def compare2model_del_designated_fold(key: str, exceptions: List, trans: List, lstm: List):
    tscore = []
    # tfirst_score=[]
    lscore = []
    # lfirst_score = []
    ct = 0
    for f in range(config.fold):
        if f in exceptions:
            continue
        trans_res = trans["all_results"][f]["result"]
        lstm_res = lstm["all_results"][f]["result"]
        if "f" not in key:
            tscore.append(trans_res[:5])
            lscore.append(lstm_res[:5])
        else:
            tscore.append(trans_res[5:])
            lscore.append(lstm_res[5:])

        ct += 1

    # 汇总
    tscore = np.array(tscore)
    lscore = np.array(lscore)

    tscore_mean = np.sum(tscore, axis=0) / ct
    lscore_mean = np.sum(lscore, axis=0) / ct

    cols = ["auc", "f1", "recall", "precision", "acc"]

    prefix = "f_" if "f" in key else ""

    cols = ["auc", "f1", "recall", "precision", "acc"]
    cols = [prefix + s for s in cols]

    figs, axes = plt.subplots()

    x = np.arange(len(cols))  # the label locations
    width = 0.35  # the width of the bars

    rects1 = axes.bar(x - width / 2, tscore_mean, width, label='trans_mean')
    rects2 = axes.bar(x + width / 2, lscore_mean, width, label='lstm_mean')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    axes.set_ylabel('scores')
    axes.set_title(f'AVG mean comparison {key}')
    axes.set_xticks(x)
    axes.set_xticklabels(cols)
    precision = 0.05
    axes.set_yticks(np.arange(0, 1 + precision, precision))
    axes.legend()
    fname = os.path.join(config.comparison_save_path, f"MEAN_{key}.png")
    figs.savefig(fname)


# delete_designated_fold("first_mean",[2],trans_info,lstm_info)
# delete_designated_fold("mean",[2],trans_info,lstm_info)
all_metrics = ["auc", "f1", "recall", "precision", "acc"]
first_metrics = ["f_" + s for s in all_metrics]


def create_table_1_all_assignments(metric, dir_pattern, model_list):
    all_metrics = ["auc", "f1", "recall", "precision", "acc"]
    first_metrics = ["f_" + s for s in all_metrics]
    figs, ax = plt.subplots()
    # get designated metric from files
    metric_key = None
    if "f_" in metric:
        metric_idx = first_metrics.index(metric)
        metric_key = "first_avg"
    else:
        metric_idx = all_metrics.index(metric)
        metric_key = "all_avg"

    assert metric_idx != -1

    x = np.arange(5)  # the label locations
    # print(os.getcwd())
    dirs = os.listdir(config.structured_data_root)
    fit_dirs = []
    for d in dirs:
        if dir_pattern in d:
            fit_dirs.append(d)

    ass2files = {}
    for d in fit_dirs:
        # key: assignment id ,value json obj
        files = os.listdir(os.path.join(config.structured_data_root, d))
        files = [os.path.join(config.structured_data_root, d, f) for f in files]
        name2json = {}

        # 用model list参数和 files匹配
        for model_name in model_list:
            for jf in files:
                # 若模型名在文件名中
                if model_name in jf:
                    with open(jf, "r") as jfp:
                        model_json = json.load(jfp)
                    name2json[model_name] = model_json
        ass2files[d[:3]] = name2json

    # final_data
    # key  :  model_name
    # value : 需要的指标在所有任务上的值，为一个dict key assignment id,value
    final_data = {}

    # 记录遍历顺序 作图最好统一
    ass_iter_order = []
    for k, name2js in ass2files.items():
        ass_iter_order.append(k)
        for m_name, m_json in name2js.items():
            if m_name not in final_data:
                final_data[m_name] = []
            ddd = m_json[metric_key][metric_idx]
            ttt = f"{ddd * 100:.2f}%"
            final_data[m_name].append(ttt)

    # 创建一个空的 DataFrame
    columns = ass_iter_order
    df = pd.DataFrame(columns=columns)
    for i, (k, v) in enumerate(final_data.items()):
        # if k=="BCQ":
        #     k="bert_question_transformer"
        df.loc[k] = v

    df.to_csv(os.path.join(config.cmp_of_all_assignment, f"{metric}_TABLE_of_all_ass.csv"), index=True)


def compare_N_model_on_one_metric_of_all_assignments(metric, dir_pattern, model_list):
    all_metrics = ["auc", "f1", "recall", "precision", "acc"]
    first_metrics = ["f_" + s for s in all_metrics]
    figs, ax = plt.subplots()
    # get designated metric from files
    metric_key = None
    if "f_" in metric:
        metric_idx = first_metrics.index(metric)
        metric_key = "first_avg"
    else:
        metric_idx = all_metrics.index(metric)
        metric_key = "all_avg"

    assert metric_idx != -1

    x = np.arange(5)  # the label locations
    # print(os.getcwd())
    dirs = os.listdir(config.structured_data_root)
    fit_dirs = []
    for d in dirs:
        if dir_pattern in d:
            fit_dirs.append(d)

    ass2files = {}
    for d in fit_dirs:
        # key: assignment id ,value json obj
        files = os.listdir(os.path.join(config.structured_data_root, d))
        files = [os.path.join(config.structured_data_root, d, f) for f in files]
        name2json = {}

        # 用model list参数和 files匹配
        for model_name in model_list:
            for jf in files:
                # 若模型名在文件名中
                if model_name in jf:
                    with open(jf, "r") as jfp:
                        model_json = json.load(jfp)
                    name2json[model_name] = model_json
        ass2files[d[:3]] = name2json

    # final_data
    # key  :  model_name
    # value : 需要的指标在所有任务上的值，为一个dict key assignment id,value
    final_data = {}

    # 记录遍历顺序 作图最好统一
    ass_iter_order = []
    for k, name2js in ass2files.items():
        ass_iter_order.append(k)
        for m_name, m_json in name2js.items():
            if m_name not in final_data:
                final_data[m_name] = []
            final_data[m_name].append(m_json[metric_key][metric_idx])

    model_num = len(model_list)
    width = 0.2
    bars = []

    for i, (model_name, Y) in enumerate(final_data.items()):
        # 中心点位置
        color = None
        if model_name in COLOR_MAP:
            color = COLOR_MAP[model_name]

        # 每个append的bar是对应model在指定指标上的一组数据，也就是在所有assignment上的数据list
        bars.append(
            ax.bar(x - (model_num * width) / 2 + ((2 * i + 1) * width / 2), Y, width, label=model_name, color=color))
    # bars中的每个item是一个model对应的在所有指标室内上的bar

    # flatted_bars中的bar是按照从左到右的顺序提取出来的bar，
    # 也就是说，bars是二维，flatted_bars是一维

    # =================================================
    # flatted_bars = []
    # for i in range(len(x)):
    #     for bar in bars:
    #         flatted_bars.append(bar[i])
    #
    # for bar in flatted_bars:
    #     height = bar.get_height()
    #     print(height)
    #     ax.annotate(f'{height:.3f}',
    #                 xy=(bar.get_x() + bar.get_width() / 2, height),
    #                 xytext=(0, 3),  # 3 points vertical offset for better readability
    #                 textcoords="offset points",
    #                 ha='center', va='bottom')
    # ========================================================
    flatted_bars = []
    for i in range(len(x)):
        for bar in bars:
            flatted_bars.append(bar[i])
    last_heights = [0] * len(x)
    for bar in flatted_bars:
        height = bar.get_height()
        print(height)
        # 检查当前柱状条的高度和相邻柱状条的高度差
        idx = list(bar.datavalues).index(height)  # 获取当前柱状条的索引
        offset = 3
        if abs(height - last_heights[idx]) < 0.05:  # 如果高度差较小，增加偏移量
            offset = 7

        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset),  # 动态调整偏移量
                    textcoords="offset points",
                    ha='center', va='bottom')
        last_heights[idx] = height  # 更新最后的高度记录

    ax.set_ylabel('scores')
    ax.set_title(f'{metric} comparison on all assignments ')
    ax.set_xticks(x)
    ax.set_xticklabels(ass_iter_order)
    precision = 0.05
    ax.set_yticks(np.arange(0, 1 + precision, precision))
    ax.legend()
    # save_dir_path=os.path.join(config.comparison_save_path,metric)
    # if not os.path.exists(save_dir_path):
    #     os.mkdir(save_dir_path)

    fname = os.path.join(config.cmp_of_all_assignment, f"{metric}_of_all_assignment.png")
    figs.savefig(fname)


if __name__ == "__main__":
    cols = ["auc", "f1", "recall", "precision", "acc"]
    table_result_for_all_assignments()
    # prefix = "f_"
    # cols = [prefix + s for s in cols]
    # for metric in cols:
    #     # compare_N_model_on_one_metric_of_all_assignments(metric, config.test_subfix,
    #     #                                                  ["lstm", "RawTrans", "BQMOD", "BCQ"])
    #     create_table_1_all_assignments(metric, config.test_subfix,
    #                                                      ["lstm", "RawTrans", "BQMOD", "BCQ"])

    # res = get_json_info_fromNfiles(["lstm", "RawTrans", "BQMOD", "BCQ"])
    # get_all_metrics_on_one_assignment(res)

    # generate_model_results_table(output_file="src/comparisons/711AAT-1005LA-BE110_SEQ[50]-487_!TGT_MATCH_SRC[60-4]3MODELS/model_cmp.csv",directory_path="src/structured_results/703TEST-BE80_SEQ[35]487_!TGT_MATCH_SRC[60-4]3MODELS")
