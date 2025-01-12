import numpy as np
import itertools
import pandas as pd
import torch
import json
from config import *
from datetime import datetime


def create_word_index_table(vocab):
    """
    Creating word to index table
    Input:
    vocab: list. The list of the node vocabulary

    """
    ixtoword = {}
    # period at the end of the sentence. make first dimension be end token
    ixtoword[0] = 'END'
    ixtoword[1] = 'UNK'
    wordtoix = {}
    wordtoix['END'] = 0
    wordtoix['UNK'] = 1
    ix = 2
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1
    return wordtoix, ixtoword


def convert_to_idx(sample, node_word_index, path_word_index):
    """
    Converting to the index 
    Input:
    sample: list. One single training sample, which is a code, represented as a list of neighborhoods.
    node_word_index: dict. The node to word index dictionary.
    path_word_index: dict. The path to word index dictionary.

    """
    sample_index = []
    for line in sample:
        components = line.split(config.single_path_sep)
        assert len(components) == 3
        if components[0] in node_word_index:
            starting_node = node_word_index[components[0]]
        else:
            starting_node = node_word_index['UNK']
        if components[1] in path_word_index:
            path = path_word_index[components[1]]
        else:
            path = path_word_index['UNK']
        if components[2] in node_word_index:
            ending_node = node_word_index[components[2]]
        else:
            ending_node = node_word_index['UNK']

        sample_index.append([starting_node, path, ending_node])
    return sample_index


MAX_CODE_LEN = 100


class data_reader():
    def __init__(self, train_path, val_path, test_path, maxstep, numofques):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.maxstep = maxstep
        self.numofques = numofques
        self.device = "cuda"  # for GPU usage or "cpu" for CPU usage

        self.bert_questions = torch.tensor(np.load(cfg.questions_BERT_embedding_file_name), device=self.device)

    def get_data(self, file_path):
        # TODO  研究数据处理
        # print(f"get data from {file_path}")
        data = []
        # label_path仅仅包含assignment439的item

        code_df = pd.read_csv(os.path.join(config.processed_data_dir, "labeled_paths.tsv"), sep="\t")
        training_students = np.load(os.path.join(config.processed_data_dir, "training_students.npy"), allow_pickle=True)
        # all_training_code 用code生成的用@链接的paths series
        # 这里检查了是否在 随机划分后的training_students.npy里

        all_training_code = code_df[code_df['SubjectID'].isin(training_students)]['RawASTPath']
        #
        separated_code = []
        for code in all_training_code:
            # 如果代码不可编译，这里code是nan，type(code)是float，也就是if不通过
            if type(code) == str:
                separated_code.append(code.split(config.paths_sep))

        node_hist = {}
        path_hist = {}
        for paths in separated_code:
            starting_nodes = []
            ending_nodes = []
            path = []
            for p in paths:
                ttt = p.split(config.single_path_sep)
                [p, s, e] = ttt
                assert len(ttt) == 3
                path.append(p)
                starting_nodes.append(s)
                ending_nodes.append(e)
            nodes = starting_nodes + ending_nodes
            for n in nodes:
                if not n in node_hist:
                    node_hist[n] = 1
                else:
                    node_hist[n] += 1
            for p in path:
                if not p in path_hist:
                    path_hist[p] = 1
                else:
                    path_hist[p] += 1
        # 统计start_tokens,end_tokens,path的词频(token和path均有可能重复,
        # 后续可以看看hash func可能有设计，这一步应该是为了建立词向量之类的NLP预处理
        node_count = len(node_hist)
        path_count = len(path_hist)
        np.save(os.path.join(config.processed_data_dir, "np_counts.npy"), [node_count, path_count])

        # small frequency then abandon, for node and path
        # 扔掉低频？但是没看到处理频率的代码
        valid_node = [node for node, count in node_hist.items()]
        valid_path = [path for path, count in path_hist.items()]

        # create ixtoword and wordtoix lists
        # 给valid_path valid_node 建立编号和互相映射，注意END 和 UNK 两个特殊标记idx为0,1
        node_word_index, node_index_word = create_word_index_table(valid_node)
        path_word_index, path_index_word = create_word_index_table(valid_path)
        kc_map_file = config.kc_question_to_kc_idx_path.format(dir=config.kc_KC_dir,
                                                               ass=Config.ASSIGNMENT_ID)
        
        with open(kc_map_file, 'r') as f:
            kc_map_data = json.load(f)
        with open(cfg.rawId2idx_json_file, "r") as ff:
            rawId2idx = json.load(ff)
        with open(cfg.idx2rawId_json_file, "r") as f2:
            idx2rawid = json.load(f2)

        with open(file_path, 'r') as file:
            #  带下标的那个文件
            print(f"open file {file_path}")
            # 每次迭代处理一个学生(对应的序列)
            for lent, css, ques, ans, asses in itertools.zip_longest(*[file] * 5):
                # 每次read一行
                # lent : length,一个学生的答题序列（多道题）的长度
                # ques  : 题号序列（从0开始)
                # ans :答案序列（是否答对)
                # css : 代码片段id，在code那张表里
                lent = int(lent.strip().strip(','))
                ques = [int(q) for q in ques.strip().strip(',').split(',')]
                ans = [int(a) for a in ans.strip().strip(',').split(',')]
                css = [cs for cs in css.strip().strip(',').split(',')]
                assignments_line = [ass for ass in asses.strip().strip(',').split(',')]
                # 把代码信息（word idx）和之前的题号以及正确与否信息并到一个数组里
                # temp是一个学生序列
                # shape=[self.maxstep, 2 * self.numofques+MAX_CODE_LEN*3]

                # 把ass raw id转为idx
                assignments_line = [config.ass2idx[int(ass)] for ass in assignments_line]
                # 把代码信息（word idx）和之前的题号以及正确与否信息并到一个数组里
                # temp是一个学生序列
                # shape=[self.maxstep, 2 * self.numofques+MAX_CODE_LEN*3]
                temp = np.zeros(shape=[self.maxstep,
                                       KCA_config.for_readdata])  # Skill DKT #1, original
                # config.questions * 2 + MAX_CODE_LEN * 3 + BCQ_config.d_bert])  # Skill DKT #1, original
                if lent >= self.maxstep:
                    steps = self.maxstep
                    extra = 0
                    # 论文中提到的，序列超过长度限制截取后面的最大部分
                    ques = ques[-steps:]
                    ans = ans[-steps:]
                    css = css[-steps:]
                    assignments_line = assignments_line[-steps:]

                else:
                    # 序列长度小于maxstep,从extra=maxstep-steps 开始
                    steps = lent
                    extra = self.maxstep - steps

                # 每次迭代处理序列中每一次submit的代码
                # temp : (50,320) 对每个学生的数据进行装填
                # ques[j] 是0-9(10个问题)的题号
                # 也就是说 path信息和ques信息放在一行里了
                # temp[j+extra][ques[j]]=1 也就是对每个step的ques部分装入数据
                first_time_obj = None
                for j in range(steps):
                    quesid0_9 = ques[j]
                    bert_ques_idx = assignments_line[j] * 10 + quesid0_9
                    if config.use_all_assignments:
                        ques_pos_id = bert_ques_idx
                    else:
                        ques_pos_id = quesid0_9
                    if ans[j] == 1:
                        # 要么ques[j]
                        # 在前10个位置填
                        temp[j + extra][ques_pos_id] = 1
                    else:
                        # 要么 10+ques[j]
                        # 在前10-20个位置填
                        temp[j + extra][ques_pos_id + config.questions] = 1

                    # 特征顺序
                    # [问题id,问题,code,time feature ,ac rate feature]
                    # 增加问题特征
                    temp[j + extra][
                    KCA_config.code2vec_dim + KCA_config.ques_encoding
                    :KCA_config.code2vec_dim + KCA_config.d_bert
                     + KCA_config.ques_encoding]= self.bert_questions[bert_ques_idx].cpu().numpy()



                    correspond_rawid = idx2rawid[str(Config.ASSIGNMENT_ID)][str(quesid0_9)]
                    kc_list_cur_ques = kc_map_data[correspond_rawid]

                    # 添加知识点 KC
                    watch_var=temp[j + extra][KCA_config.for_readdata - KCA_config.KC_one_hot_dim:]
                    temp[j + extra][KCA_config.for_readdata - KCA_config.KC_one_hot_dim:][kc_list_cur_ques] = 1

                    # bug label中没有序号那个文件里有的codeid
                    # label是由所有code生成的吗，如果是，就是数据集问题
                    # 如果label是部分，那就是划分问题？
                    j_ = code_df[code_df['CodeStateID'] == int(css[j])]
                    code = j_['RawASTPath'].iloc[0]
                    # code = code_df[code_df['CodeStateID'] == int(css[j])]['RawASTPath'].iloc[0]
                    code_state = j_['compile_state'].iloc[0]
                    raw_score = j_['raw_score'].iloc[0]
                    raw_time = j_["ServerTimestamp"].iloc[0]
                    # -------------把path处理成词表中对应的id，没有做嵌入------------
                    # date_string = "2019-03-09T03:09:52"
                    date_time_obj = datetime.strptime(raw_time, '%Y-%m-%dT%H:%M:%S')
                    if j == 0:
                        # 在第一次循环时，记录下第一轮的时间
                        first_time_obj = date_time_obj

                        # 计算当前时间与第一轮时间的差异（单位为分钟）
                    time_diff = (date_time_obj - first_time_obj).total_seconds() / 60.0
                    # time_offsets.append(time_diff)
                    # 增加10位
                    # 先按照字典序排序
                    # date_list = [date_time_obj.year, date_time_obj.month, date_time_obj.day, date_time_obj.hour,
                    #              date_time_obj.minute, date_time_obj.second]


                    # 先复制几份吧
                    # 时间偏移，复制10份，

                    time_feature=[time_diff*KCA_config.time_encoding]
                    temp[j + extra][KCA_config.code2vec_dim + KCA_config.d_bert + KCA_config.ques_encoding:
                    KCA_config.for_readdata-KCA_config.score_encoding-KCA_config.KC_one_hot_dim]=time_diff

                    #
                    #
                    # time_list = [date_time_obj.month, date_time_obj.day, date_time_obj.hour,
                    #              date_time_obj.minute, date_time_obj.second]
                    #
                    # def fill_time_in(t, i, a):
                    #     if t < 10:
                    #         a[i + 1] = t
                    #     else:
                    #         a[i] = t // 10
                    #         a[i + 1] = t % 10
                    #
                    # # Time
                    # for i, t in enumerate(time_list):
                    #     # i*2是遍历 YY MM DD HH MM SS
                    #     fill_time_in(t,
                    #                  KCA_config.code2vec_dim + KCA_config.d_bert + KCA_config.ques_encoding + i * 2,
                    #                  temp[j + extra])

                    # AC rate
                    raw_score_dup=[raw_score*KCA_config.score_encoding]
                    temp[j + extra][
                    KCA_config.for_readdata - KCA_config.score_encoding-KCA_config.KC_one_hot_dim:
                    KCA_config.for_readdata - KCA_config.KC_one_hot_dim] = raw_score_dup

                    # if条件不通过,nan,即代码不可编译，path为空时，
                    # 由于temp初始化用np.zeroes，所以填全0

                    if type(code) == str:
                        # 该代码片段中所有符合条件的path
                        code_paths = code.split(config.paths_sep)
                        raw_features = convert_to_idx(code_paths, node_word_index, path_word_index)
                        if len(raw_features) < MAX_CODE_LEN:
                            # path数量不足，填充[0,0,0]对
                            raw_features += [[0, 0, 0]] * (MAX_CODE_LEN - len(raw_features))
                        else:
                            # path数量超过100（MAX_CODE_LEN），截取100条
                            raw_features = raw_features[:MAX_CODE_LEN]
                        # raw feature (100,3) -----reshape---->   features (1,300)
                        features = np.array(raw_features).reshape(-1, MAX_CODE_LEN * 3)

                        # 把代码信息（word idx）和之前的ques题号以及ques正确与否信息并到一个数组里
                        temp[j + extra][KCA_config.ques_encoding
                                        :KCA_config.ques_encoding + KCA_config.code2vec_dim] = features

                        # temp shape (50,320)
                # 一个学生序列处理完毕(temp),加入data
                data.append(temp.tolist())
            # data   List---->np.array
            print('done: ' + str(np.array(data).shape))

        # 返回一个文件对应的所有学生数据
        return data

    def get_train_data(self):
        print('loading train data...')
        train_data = self.get_data(self.train_path)
        val_data = self.get_data(self.val_path)
        return np.array(train_data + val_data)

    def get_test_data(self):
        print('loading test data...')
        test_data = self.get_data(self.test_path)
        return np.array(test_data)

if __name__=="__main__":
    fold=1
    # node_count, path_count = np.load(os.path.join(config.processed_data_dir, "np_counts.npy"))

    handle = data_reader(os.path.join(cfg.splited_data_dir, "train_firstatt_" + str(fold) + ".csv"),
                         os.path.join(cfg.splited_data_dir, "val_firstatt_" + str(fold) + ".csv"),
                         os.path.join(cfg.splited_data_dir, "test_data.csv"), config.length,
                         config.questions)
    handle.get_train_data()
