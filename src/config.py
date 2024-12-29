import math
import os.path
import torch

print("测试是否sync成功")


# 需要是64的倍数
# 1240 + 768
def ROUNDUP(x):
    if x % 64 == 0:
        return x
    i = x // 64
    return 64 * (i + 1)


class MySingleton(object):
    def __init__(self, cls):
        self._cls = cls
        self._instance = {}

    def __call__(self):
        if self._cls not in self._instance:
            self._instance[self._cls] = self._cls()
        return self._instance[self._cls]


RUN_TIMES = 10
# RUN_TIMES = 10

BCQ_EPOCH = 55
BCQ_LAYER = 3

KCA_EPOCH = 40





BCQL_EPOCH = 80
BCQL_LAYER = 3

BLAA_EPOCH=40

L_EPOCH = 40
# L_EPOCH = 1

BQMOD_EPOCH = 80
BQMOD_LAYER = 4

Q_LAYER = 4
Q_EPOCH = 1

R_LAYER = 3
R_EPOCH = 50  # 50效果最好

CBTQ_EPOCH = 60
CBTQ_LAYER = 3

# CT5Q_EPOCH = 60
# CT5Q_EPOCH = 50
# CT5Q_EPOCH = 60
CT5Q_EPOCH = 60
CT5Q_LAYER = 4

CT5QER_EPOCH = 100
CT5QER_LAYER = 6

BQ_LAYER = 4
BQ_EPOCH = 1
# PREFIX = f"BQ[{BQ_LAYER}]{BQ_EPOCH}-Q[{Q_LAYER}]{Q_EPOCH}-R[{R_LAYER}]{R_EPOCH}-BQMOD[{BQMOD_LAYER}]{BQMOD_EPOCH}"
# PREFIX = f"R{R_EPOCH}[{R_LAYER}]_BCQ{BCQ_EPOCH}[{BCQ_LAYER}]"


CT5LSTM_EPOCH = 35
CT5LSTM_LAYER = 1  # USELESS

PREFIX = f"!TGT_MATCH_SRC[{CT5Q_EPOCH}-{CT5Q_LAYER}]"

DEVICE = None
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


class BertRawConfig:
    def __init__(self):
        # self.before_trans_alignment_d = 340
        # self.after_trans_alignment_d_model = 640
        # self.d_bert = 768
        self.before_trans_alignment_d = 788  # 768+20
        self.codebert_transformer_d = 768
        self.raw_trans_epochs = R_EPOCH
        self.num_decoder_layers = R_LAYER
        self.num_encoder_layers = R_LAYER


# @MySingleton
class Config:
    # Assignment
    # 492 A5 79.14%      781        0.15050 0  0
    # 494 A1 72.75%      745        0.15050 0  0
    # 439 A4 74.31%      745        0.15050 0  0
    # 502 A3 80.40%      792        0.15050 0  0
    # 487 A2 76.56%                   0.15050 0   0.15050
    ASSIGNMENT_ID = 492

    MODEL_TYPE = "QuesT"

    def __init__(self):
        # "../huggingface_cloned/codebert-base"

        # 2-16

        self.assignments = [439, 487, 492, 494, 502]
        # 别动
        self.ass2idx = {439: 0, 487: 1, 492: 2, 494: 3, 502: 4}

        self.frontend_local_model = "../huggingface_cloned/codet5p-110m-embedding"

        self.paths_sep = "@#$"
        self.use_all_assignments = False
        self.single_path_sep = "!#!"

        self.frontend = "code2vec"
        self.dataset = "codeDKT_dataset"
        self.d_model_no_ques = 320  # just for transformer

        # TODO 每加一个模型。更新run文件中的model_type以及添加
        # self.all_models=["QuesTrans","lstm","RawTrans"]

        self.length = 50
        # self.length = 35
        # self.length = 40
        if self.use_all_assignments:
            self.questions = 50
        else:
            self.questions = 10
        self.lr = 0.0005

        self.bs = 32  # batch_size
        # self.bs = 64  # batch_size
        # self.bs =16  # batch_size

        # self.bs = 128
        # self.epochs = 40
        # self.lstm_epochs = 40
        # self.raw_trans_epochs = 40
        # self.ques_trans_epochs = 40
        self.hidden = 128
        self.layers = 1
        self.fold = 10
        self.word_embedding_size = 100
        self.ceiling = 100

        # self.assignment = 439
        # 每个不同的assignment创造一个目录
        # self.assignment = 439
        self.code_path_length = 8
        self.code_path_width = 2
        self.raw_data_dir = "../data"
        self.use_all_dataset_prefix = "all_ASS"
        # ----------------------------------------------------------------------
        # 知识点
        self.kc_related="./KC_related"
        self.kc_v2idx_dir = f"{self.kc_related}/value2idx"
        self.kc_idx2v_dir = f"{self.kc_related}/idx2value"
        self.kc_KC_dir = f"{self.kc_related}/KC"

        self.kc_value_to_idx_path = '{dir}/value_to_idx_{ass}.json'
        self.kc_idx_to_value_path = '{dir}/idx_to_value_{ass}.json'
        self.kc_question_to_kc_idx_path = '{dir}/question_to_kc_idx_{ass}.json'

        self.kc_processed_main_table=f'{self.kc_related}/202406ass_qid_qtext_add_KC_idx.xlsx'


        blaa_lr_num = 4
        lamda = 0.1
        # ----------------------------------------------------------------------

        # 分开控制，比如epoch变了，但是preprocess不用重新生产了，可以控制数据集一样
        # 方便比较结果

        # ----------------------------------------------------------------------
        # TEMP  临时变量   不改代码结构了 手动监控
        dir_lr_num = int(QPALA_lr*10000)
        lamda = 0.1
        self.model_to_run = ["QPALA","LSTM"]
        # ----------------------------------------------------------------------

        # self.preprocess_dir_prefix=f"0703_{Config.ASSIGNMENT_ID}"
        # self.preprocess_dir_prefix = f"0712_{Config.ASSIGNMENT_ID}"
        self.preprocess_dir_prefix = f"0722fix_{Config.ASSIGNMENT_ID}"

        # 60其实是40 加一个F区分

        # 0.15
        # 40 40  0.005
        # 50 50  0.008
        # e50 s40  0.005
        # e55 s45  0.005/0.007
        #0.1 50 50
        self.after_training_dir_prefix = (f"810FF{self.model_to_run[0]}-"
                                          f"BE{QPALA_EPOCH}-"
                                          f"lr{dir_lr_num}-"
                                          f"la{QPALA_attn_lamda}-"
                                          f"SEQ[{self.length}]")  # optional 为了在目录堆里能快速找到 平时建议设为空


        self.test_subfix = "3MODELS"

        self.post_preprocess_main_table = os.path.join(self.raw_data_dir, "MainTable_add_col.csv")

        self.raw_main_table = os.path.join(self.raw_data_dir, "MainTable.csv")
        self.raw_code_table = os.path.join(self.raw_data_dir, "CodeStates.csv")
        # self.chosed_main_table=
        # 删去了经过tokenizer后超过512的数据
        self.raw_main_table_for_BERT = os.path.join(self.raw_data_dir, "MainTable_For_BERT.csv")
        self.raw_main_table_for_codeT5 = os.path.join(self.raw_data_dir, "MainTable_For_codeT5.csv")
        self.all_large_code = os.path.join(self.raw_data_dir, "all_large_code.csv")
        # self.test_subfix = "_0216_SINGLE"
        if not self.use_all_assignments:
            self.processed_data_dir = f"../processed_data/{self.preprocess_dir_prefix}_{self.test_subfix}"
        else:
            self.processed_data_dir = f"../processed_data/{self.use_all_dataset_prefix}_{self.test_subfix}"
        # preprocess 不影响
        self.rawId2idx_json_file = os.path.join(self.processed_data_dir, "rawId2idx.json")
        self.idx2rawId_json_file = os.path.join(self.processed_data_dir, "idx2rawId.json")

        self.all_question_dir = "question_root"
        self.all_text_questions_file = os.path.join(self.all_question_dir, "all_questions.json")

        self.question_dir = os.path.join(self.processed_data_dir, "questions")
        self.codeID2codeBERT_vec = os.path.join(self.processed_data_dir, "codeid_codevec.npy")

        # TODO handle questions
        self.question_file_name = os.path.join(self.question_dir, f"questions{Config.ASSIGNMENT_ID}.json")

        # 真正id(e.g. 234)---->序号(i.e. 0-9 10个)
        self.question_id_map_fname = os.path.join(self.question_dir, f"question_id_map.json")

        # for BERT pretrained embedding
        self.questions_BERT_embedding_file_name = os.path.join(self.question_dir, "questions_embedding.npy")
        # For just word2id
        self.most_common_word_num = 50
        self.question_len = 100
        self.question_word2id_map_fname = os.path.join(self.question_dir, "question_word2id.json")
        self.question_word2id_file_name = os.path.join(self.question_dir, "id_questions.npy")
        if not os.path.exists(self.question_dir):
            os.makedirs(self.question_dir)
        with open(os.path.join(self.question_dir, "for_sync.txt"), "w") as f:
            print("okoko")
            f.write("hello")
            print(f"name of questions file should be :::: {self.question_file_name}")

        self.splited_data_dir = os.path.join(self.processed_data_dir, "train_val_test")
        if not os.path.exists(self.splited_data_dir):
            os.makedirs(self.splited_data_dir, exist_ok=True)

        # switch
        # self.model_type = "lstm"
        # SAVE MODEL
        self.save_path = "../models"
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.unique_params_mark = PREFIX
        self.structured_data_root = "1new_structured_results"
        self.all_comparison_root = "1all_comparisons"
        # self.structured_data_root = "structured_results"
        if not self.use_all_assignments:

            self.json_dir = f"{self.structured_data_root}/{self.after_training_dir_prefix}{Config.ASSIGNMENT_ID}_{self.unique_params_mark}{self.test_subfix}"
        else:
            self.json_dir = f"{self.structured_data_root}/{self.use_all_dataset_prefix}_{self.unique_params_mark}{self.test_subfix}"

        if not os.path.exists(self.json_dir):
            os.makedirs(self.json_dir, exist_ok=True)
        if not self.use_all_assignments:
            self.loss_dir = f"losses/{self.after_training_dir_prefix}{Config.ASSIGNMENT_ID}_{self.test_subfix}"
        else:
            self.loss_dir = f"losses/{self.use_all_dataset_prefix}_{self.test_subfix}"

        if not os.path.exists(self.loss_dir):
            os.makedirs(self.loss_dir, exist_ok=True)

        if not self.use_all_assignments:
            self.pred_dir = f"preds/{self.after_training_dir_prefix}{Config.ASSIGNMENT_ID}_{self.unique_params_mark}{self.test_subfix}"
        else:
            self.pred_dir = f"preds/{self.use_all_dataset_prefix}_{self.unique_params_mark}{self.test_subfix}"

        if not os.path.exists(self.pred_dir):
            os.makedirs(self.pred_dir, exist_ok=True)

        # PLACE为占位符，在不同模型运行时被分别替换
        # 每次跑都重写文件，OK，可以覆盖
        if not self.use_all_assignments:
            self.json_file_name = f"{self.dataset}_PLACE_ass{Config.ASSIGNMENT_ID}_EPOCH_HHH.json"
        else:
            self.json_file_name = f"{self.dataset}_PLACE_ALL_ASS_EPOCH_HHH.json"

        self.comparison_root = "comparisons"  #也是table的目录
        self.comparison_table_file_name=os.path.join(self.comparison_root,"table.csv")
        # self.blend_of_all_assignment=
        # 横向比较
        self.cmp_of_all_assignment = os.path.join(self.comparison_root,
                                                  f"{self.use_all_dataset_prefix}_included_{self.test_subfix}")

        # 不分assignment比较
        if not self.use_all_assignments:
            self.comparison_save_path = f"{self.comparison_root}/{self.after_training_dir_prefix}-{Config.ASSIGNMENT_ID}_{self.unique_params_mark}{self.test_subfix}"
        else:
            self.comparison_save_path = f"{self.comparison_root}/{self.use_all_dataset_prefix}_{self.unique_params_mark}{self.test_subfix}"
        self.all_ass_table_compare_dir=self.comparison_save_path = f"{self.all_comparison_root}/ALL{self.after_training_dir_prefix}"

        # self.comparison_save_path = f"{self.comparison_root}/{Config.ASSIGNMENT_ID}_{self.unique_params_mark}{self.test_subfix}"
        self.comparison_per_fold = os.path.join(self.comparison_save_path, "per_fold")
        self.comparison_avg_no_del = os.path.join(self.comparison_save_path, "avg_no_del")
        self.comparison_del_some_mean = os.path.join(self.comparison_save_path, "del_some_mean")
        # self.single_metric_comparison=os.path.join(self.comparison_save_path,)
        # compare文件也每次重写
        if not os.path.exists(self.cmp_of_all_assignment):
            os.makedirs(self.cmp_of_all_assignment, exist_ok=True)

        if not os.path.exists(self.comparison_per_fold):
            os.makedirs(self.comparison_per_fold, exist_ok=True)

        if not os.path.exists(self.comparison_avg_no_del):
            os.makedirs(self.comparison_avg_no_del, exist_ok=True)

        if not os.path.exists(self.comparison_del_some_mean):
            os.makedirs(self.comparison_del_some_mean, exist_ok=True)
        if not os.path.exists(self.all_ass_table_compare_dir):
            os.makedirs(self.comparison_del_some_mean, exist_ok=True)

class LSTMConfig:
    def __init__(self):
        self.lstm_epochs = L_EPOCH

    def __str__(self):
        return "&&&&&&&&&&&&&&&&&&&&&&&\n\n我是RawTransformer的配置类\n\n&&&&&&&&&&&&&&&&&&&&&&&"


class CT5Q_No_time_Config:
    """
    CodeT5_BertQuestion_TransformerConfig     with no time feature
    """

    def __init__(self, sconfig=None):
        self.config = sconfig
        self.bert_ques_dim = 768
        self.codeT5_embed_dim = 256

        self.transformed_code = 256

        self.transformed_question = 100

        # TODO check
        if self.config.use_all_assignments:
            self.ques_encoding = 100
        else:
            self.ques_encoding = 20
        self.score_encoding = 10

        # self.time_encoding = 10

        self.before_trans_alignment_d = \
            self.transformed_code + self.transformed_question + \
            self.ques_encoding + self.score_encoding

        self.after_trans_alignment_d_model = ROUNDUP(self.before_trans_alignment_d)
        self.d_bert = 768
        self.raw_trans_epochs = CT5Q_EPOCH
        self.num_decoder_layers = CT5Q_LAYER
        self.num_encoder_layers = CT5Q_LAYER
        self.ques_trans_epochs = CT5Q_EPOCH
        print(f"model : CT5Q args : \n  before dim : {self.before_trans_alignment_d}   \n"
              f"after dim : {self.after_trans_alignment_d_model}")


class CodeT5_BertQuestion_TransformerConfig:
    """
    CodeBert_BertQuestion_Transformer  CBTQ or CBT

    # 特征顺序
                    # [问题id,问题,code,time feature ,ac rate feature]
                    # 增加问题特征


    """

    def __init__(self, sconfig=None):
        self.config = sconfig
        self.bert_ques_dim = 768
        self.codeT5_embed_dim = 256
        self.lr = 0.0002
        self.transformed_code = 256

        self.transformed_question = 100

        # TODO check
        if self.config.use_all_assignments:
            self.ques_encoding = 100
        else:
            self.ques_encoding = 20
        self.score_encoding = 10

        self.time_encoding = 10

        self.before_trans_alignment_d = \
            self.transformed_code + self.transformed_question + \
            self.ques_encoding + self.score_encoding + self.time_encoding

        self.after_trans_alignment_d_model = ROUNDUP(self.before_trans_alignment_d)
        self.d_bert = 768
        self.raw_trans_epochs = CT5Q_EPOCH
        self.num_decoder_layers = CT5Q_LAYER
        self.num_encoder_layers = CT5Q_LAYER
        self.ques_trans_epochs = CT5Q_EPOCH
        print(f"model : CT5Q args : \n  before dim : {self.before_trans_alignment_d}   \n"
              f"after dim : {self.after_trans_alignment_d_model}")


class CT5LSTM:
    """
        CodeBert_BertQuestion_Transformer_Exercise_Response  CT5QER

        # 特征顺序
                        # [问题id,问题,code,time feature ,ac rate feature]
                        # 增加问题特征
        """

    def __init__(self, sconfig=None):
        self.config = sconfig
        self.bert_ques_dim = 768
        self.codeT5_embed_dim = 256

        self.transformed_code = 256

        self.transformed_question = 100

        # TODO check
        if self.config.use_all_assignments:
            self.ques_encoding = 100
        else:
            self.ques_encoding = 20
        self.score_encoding = 10

        self.time_encoding = 10

        self.before_trans_alignment_d = \
            self.transformed_code + self.transformed_question + \
            self.ques_encoding + self.score_encoding + self.time_encoding
        # 先用before过一个线性层，这个线性层是最终transformer输入维度（64倍数）-20
        for_ts = ROUNDUP(self.before_trans_alignment_d)
        while for_ts - self.ques_encoding < self.before_trans_alignment_d:
            for_ts += 64

        self.align_without_onehot_ques = for_ts - self.ques_encoding
        # 然后再接上一个线性层
        self.after_trans_alignment_d_model = for_ts

        # 这样修改之后，只需在原始代码上加一个concat即可

        self.d_bert = 768
        self.raw_trans_epochs = CT5LSTM_EPOCH
        self.num_decoder_layers = CT5LSTM_LAYER
        self.num_encoder_layers = CT5LSTM_LAYER
        self.ques_trans_epochs = CT5LSTM_EPOCH
        print(f"model : CT5Q args : \n  before dim : {self.before_trans_alignment_d}   \n"
              f"after dim : {self.after_trans_alignment_d_model}")


class CodeT5_BertQuestion_TransformerConfig_Exercise_Response:
    """
    CodeBert_BertQuestion_Transformer_Exercise_Response  CT5QER

    # 特征顺序
                    # [问题id,问题,code,time feature ,ac rate feature]
                    # 增加问题特征
    """

    def __init__(self, sconfig=None):
        self.config = sconfig
        self.bert_ques_dim = 768
        self.codeT5_embed_dim = 256
        self.lr = 0.0002
        self.transformed_code = 256

        self.transformed_question = 100

        # TODO check
        if self.config.use_all_assignments:
            self.ques_encoding = 100
        else:
            self.ques_encoding = 20
        self.score_encoding = 10

        self.time_encoding = 10

        self.before_trans_alignment_d = \
            self.transformed_code + self.transformed_question + \
            self.ques_encoding + self.score_encoding + self.time_encoding
        # 先用before过一个线性层，这个线性层是最终transformer输入维度（64倍数）-20
        for_ts = ROUNDUP(self.before_trans_alignment_d)
        while for_ts - self.ques_encoding < self.before_trans_alignment_d:
            for_ts += 64

        self.align_without_onehot_ques = for_ts - self.ques_encoding
        # 然后再接上一个线性层
        self.after_trans_alignment_d_model = for_ts

        # 这样修改之后，只需在原始代码上加一个concat即可

        self.d_bert = 768
        self.raw_trans_epochs = CT5QER_EPOCH
        self.num_decoder_layers = CT5QER_LAYER
        self.num_encoder_layers = CT5QER_LAYER
        self.ques_trans_epochs = CT5QER_EPOCH
        print(f"model : CT5Q args : \n  before dim : {self.before_trans_alignment_d}   \n"
              f"after dim : {self.after_trans_alignment_d_model}")


class CodeBert_BertQuestion_TransformerConfig:
    """
    CodeBert_BertQuestion_Transformer  CBTQ or CBT
    """

    def __init__(self, sconfig=None):
        self.config = sconfig
        self.transformed_code = 400
        self.transformed_question = 100
        self.before_trans_alignment_d = self.transformed_code + self.transformed_question + 20
        self.after_trans_alignment_d_model = ROUNDUP(self.before_trans_alignment_d)
        self.d_bert = 768
        self.raw_trans_epochs = CBTQ_EPOCH
        self.num_decoder_layers = CBTQ_LAYER
        self.num_encoder_layers = CBTQ_LAYER

    def __str__(self):
        return ("&&&&&&&&&&&&&&&&&&&&&&&\n\n"
                "我是CodeBert_BertQuestion_Transformer的配置类"
                "\n\n&&&&&&&&&&&&&&&&&&&&&&&")


class RawTransformerConfig:

    def __init__(self, config=None):
        self.config = config
        if self.config is not None:
            if self.config.use_all_assignments:
                self.before_trans_alignment_d = 500
                self.ques_encoding = 100
            else:
                self.before_trans_alignment_d = 340

                self.ques_encoding = 20
        else:
            self.before_trans_alignment_d = 340
            self.ques_encoding = 20

        self.after_trans_alignment_d_model = 640
        self.d_bert = 768
        self.raw_trans_epochs = R_EPOCH
        self.num_decoder_layers = R_LAYER
        self.num_encoder_layers = R_LAYER

    def __str__(self):
        return "&&&&&&&&&&&&&&&&&&&&&&&\n\n我是RawTransformer的配置类\n\n&&&&&&&&&&&&&&&&&&&&&&&"


class BertBaseQuesTransformerConfig:
    def __init__(self):
        self.before_trans_alignment_d = 340 + 768
        self.after_trans_alignment_d_model = 64 * 18
        self.d_bert = 768

        self.ques_trans_epochs = BQ_EPOCH

        self.num_decoder_layers = BQ_LAYER
        self.num_encoder_layers = BQ_LAYER


class BertQuesModTransformerConfig:
    def __init__(self):
        self.before_trans_alignment_d = 1240 + 768
        self.after_trans_alignment_d_model = ROUNDUP(self.before_trans_alignment_d)
        self.d_bert = 768

        self.ques_trans_epochs = BQMOD_EPOCH

        self.num_decoder_layers = BQMOD_LAYER
        self.num_encoder_layers = BQMOD_LAYER


class BertCompressedQuesTransformerConfig:
    def __init__(self, sconfig):
        self.config = sconfig
        # self.before_trans_alignment_d = 1240 + 768
        self.scale_bert = 50
        # self.scale_bert = 64

        self.before_trans_alignment_d = 340 + self.scale_bert

        if self.config.use_all_assignments:
            self.ques_encoding = 100
        else:
            self.ques_encoding = 20

        self.code2vec_dim = 300
        self.score_encoding = 10
        self.num_heads = 2

        # 如果是纯LSTM 这就是LSTM层的维度
        self.rnn_dim = self.before_trans_alignment_d

        # 先用before过一个线性层，这个线性层是最终transformer输入维度（64倍数）-20
        for_ts = ROUNDUP(self.before_trans_alignment_d)
        while for_ts - self.ques_encoding < self.before_trans_alignment_d:
            for_ts += 64

        self.align_without_onehot_ques = for_ts - self.ques_encoding
        # 然后再接上一个线性层
        self.after_trans_alignment_d_model = for_ts

        # 这样修改之后，只需在原始代码上加一个concat即可

        self.d_bert = 768  # bert question 读入时的维度
        self.for_readdata = self.code2vec_dim + self.ques_encoding + self.d_bert

        self.ques_trans_epochs = BCQL_EPOCH

        self.num_decoder_layers = BCQL_LAYER
        self.num_encoder_layers = BCQL_LAYER
        self.lr = 0.0003


class BertCompressedQuesAttentionLSTMConfig:
    """
     BCQ_LSTMA
    """

    def __init__(self, sconfig):
        self.config = sconfig
        # self.before_trans_alignment_d = 1240 + 768
        self.scale_bert = 50
        if self.config.use_all_assignments:
            self.ques_encoding = 100
        else:
            self.ques_encoding = 20
        self.score_encoding = 10
        # self.lr = 0.0003
        self.lr = 0.001 # 8-3
        self.time_encoding = 10
        self.code2vec_dim = 300
        self.d_bert = 768
        # readdata时使用
        self.for_readdata = self.code2vec_dim + self.ques_encoding + self.d_bert + self.score_encoding + self.time_encoding
        # rnn dim
        self.rnn_dim = self.code2vec_dim + 2 * self.ques_encoding + self.scale_bert + self.score_encoding + self.time_encoding
        self.before_trans_alignment_d = self.rnn_dim

        self.num_heads = 2

        # 在无self attentiono 的lstm中，不需要关注下面的内容

        # 先用before过一个线性层，这个线性层是最终transformer输入维度（64倍数）-20
        for_ts = ROUNDUP(self.before_trans_alignment_d)
        while for_ts - self.ques_encoding < self.before_trans_alignment_d:
            for_ts += 64

        self.align_without_onehot_ques = for_ts - self.ques_encoding
        # 然后再接上一个线性层
        self.after_trans_alignment_d_model = for_ts

        # 由于rnn在最后，需要ques_pos（KCLSTM中还要加KC_one_hot)
        self.final_rnn_dim=self.after_trans_alignment_d_model+self.ques_encoding

        # 这样修改之后，只需在原始代码上加一个concat即可，
        # 也就是说，ques的one-hot一定不能破坏了送进模型

        self.d_bert = 768

        self.ques_trans_epochs = BCQL_EPOCH

        self.num_decoder_layers = BCQL_LAYER
        self.num_encoder_layers = BCQL_LAYER


class BLSTM_Attention_After_config:
    """
     BLAA
    """

    def __init__(self, sconfig):
        self.config = sconfig
        self.total_attention_weight = 0.15
        # self.before_trans_alignment_d = 1240 + 768
        self.scale_bert = 50
        if self.config.use_all_assignments:
            self.ques_encoding = 100
        else:
            self.ques_encoding = 20
        self.score_encoding = 10
        # self.lr = 0.0003
        self.lr = 0.005
        self.time_encoding = 10
        self.code2vec_dim = 300
        self.d_bert = 768
        # readdata时使用
        self.for_readdata = self.code2vec_dim + self.ques_encoding + self.d_bert + self.score_encoding + self.time_encoding
        # rnn dim
        self.rnn_dim = self.code2vec_dim + 2 * self.ques_encoding + self.scale_bert + self.score_encoding + self.time_encoding
        self.before_trans_alignment_d = self.rnn_dim

        self.num_heads = 2

        # 在无self attentiono 的lstm中，不需要关注下面的内容

        # 先用before过一个线性层，这个线性层是最终transformer输入维度（64倍数）-20
        for_ts = ROUNDUP(self.before_trans_alignment_d)
        while for_ts - self.ques_encoding < self.before_trans_alignment_d:
            for_ts += 64

        self.align_without_onehot_ques = for_ts - self.ques_encoding
        # 然后再接上一个线性层
        self.after_trans_alignment_d_model = for_ts

        # 这样修改之后，只需在原始代码上加一个concat即可，
        # 也就是说，ques的one-hot一定不能破坏了送进模型

        self.d_bert = 768

        self.ques_trans_epochs = BCQL_EPOCH

        self.num_decoder_layers = BCQL_LAYER
        self.num_encoder_layers = BCQL_LAYER

class KCA_Config:
    """
     KCA
    """

    def __init__(self, sconfig):
        self.config = sconfig
        self.total_attention_weight = 0.15
        # self.before_trans_alignment_d = 1240 + 768
        self.scale_bert = 50
        if self.config.use_all_assignments:
            self.ques_encoding = 100
        else:
            self.ques_encoding = 20
        self.score_encoding = 10
        self.lr = 0.001
        # self.lr = 0.0003
        self.time_encoding = 10
        self.code2vec_dim = 300
        self.d_bert = 768
        self.KC_one_hot_dim=20
        # readdata时使用
        self.for_readdata = self.code2vec_dim + self.ques_encoding + self.d_bert + self.score_encoding + self.time_encoding + self.KC_one_hot_dim
        # rnn dim
        self.rnn_dim = self.code2vec_dim + 2 * self.ques_encoding + self.scale_bert + self.score_encoding + self.time_encoding + self.KC_one_hot_dim
        self.before_trans_alignment_d = self.rnn_dim

        self.num_heads = 2

        # 在无self attentiono 的lstm中，不需要关注下面的内容

        # 先用before过一个线性层，这个线性层是最终transformer输入维度（64倍数）-20
        for_ts = ROUNDUP(self.before_trans_alignment_d)
        while for_ts - self.ques_encoding < self.before_trans_alignment_d:
            for_ts += 64

        self.align_without_onehot_ques = for_ts - self.ques_encoding- self.KC_one_hot_dim
        # 然后再接上一个线性层
        self.after_trans_alignment_d_model = for_ts

        # 这样修改之后，只需在原始代码上加一个concat即可，
        # 也就是说，ques的one-hot一定不能破坏了送进模型

        self.d_bert = 768

        self.ques_trans_epochs = BCQL_EPOCH

        self.num_decoder_layers = BCQL_LAYER
        self.num_encoder_layers = BCQL_LAYER

class KCLSTMA_Config:
    """
     KCLSTMA


     先MultiHead ATTN 再 LSTM
    """

    def __init__(self, sconfig):
        self.config = sconfig
        self.total_attention_weight = 0.15
        # self.before_trans_alignment_d = 1240 + 768
        self.scale_bert = 50
        if self.config.use_all_assignments:
            self.ques_encoding = 100
        else:
            self.ques_encoding = 20
        self.score_encoding = 10
        self.lr = 0.0003
        self.time_encoding = 10
        self.code2vec_dim = 300
        self.d_bert = 768
        self.KC_one_hot_dim=20
        self.kc_embed_dim = 64
        # self.ques_query_64
        # readdata时使用
        self.for_readdata = (self.code2vec_dim + self.ques_encoding +
                             self.d_bert + self.score_encoding + self.time_encoding + self.KC_one_hot_dim)
        # rnn dim
        # 这里的*2 一次是code2vec path concat，一次是fit_trans前concat

        #kc只能一次
        self.before_trans_alignment_d = (self.code2vec_dim +  2*self.ques_encoding +
                        self.scale_bert + self.score_encoding + self.time_encoding + self.KC_one_hot_dim+self.kc_embed_dim)

        self.num_heads = 2

        # 在无self attentiono 的lstm中，不需要关注下面的内容

        # 先用before过一个线性层，这个线性层是最终transformer输入维度（64倍数）-20
        for_ts = ROUNDUP(self.before_trans_alignment_d)
        while for_ts - self.ques_encoding -self.KC_one_hot_dim< self.before_trans_alignment_d:
            for_ts += 64

        self.align_without_onehot_ques = for_ts - self.ques_encoding- self.KC_one_hot_dim
        # 然后再接上一个线性层
        self.after_trans_alignment_d_model = for_ts



        # 这样修改之后，只需在原始代码上加一个concat即可，
        # 也就是说，ques的one-hot一定不能破坏了送进模型

        self.rnn_dim=self.after_trans_alignment_d_model+self.KC_one_hot_dim+self.ques_encoding



        self.d_bert = 768

        self.ques_trans_epochs = BCQL_EPOCH

        self.num_decoder_layers = BCQL_LAYER
        self.num_encoder_layers = BCQL_LAYER


QPKT_EPOCH=40

class QPKT_Config:
    """
     QPKT

    path attn + question
    先用原生 LSTM
    """

    def __init__(self, sconfig):
        self.config = sconfig
        self.total_attention_weight = 0.15
        # self.before_trans_alignment_d = 1240 + 768
        self.scale_bert = 64
        if self.config.use_all_assignments:
            self.ques_encoding = 100
        else:
            self.ques_encoding = 20
        self.score_encoding = 10
        # self.lr = 0.001
        self.lr = 0.0005
        self.time_encoding = 10

        # transform了一下 不好的话后面改回去
        self.code2vec_dim = 300

        self.code2vec_trans_dim = 256


        self.d_bert = 768
        self.KC_one_hot_dim=20
        self.kc_embed_dim = 256
        # self.ques_query_64
        # readdata时使用
        self.for_readdata = (self.code2vec_dim + self.ques_encoding +
                             self.d_bert + self.score_encoding + self.time_encoding + self.KC_one_hot_dim)
        # rnn dim
        # 这里的*2 一次是code2vec path concat，一次是fit_trans前concat

        #kc只能一次 self.ques_encoding第一个被trans成256了 所以只有一个
        self.before_trans_alignment_d = (self.kc_embed_dim +  self.ques_encoding +
                        self.scale_bert + self.KC_one_hot_dim)

        self.rnn_dim=self.before_trans_alignment_d


        self.num_heads = 2

        # 在无self attentiono 的lstm中，不需要关注下面的内容

        # 先用before过一个线性层，这个线性层是最终transformer输入维度（64倍数）-20
        for_ts = ROUNDUP(self.before_trans_alignment_d)
        while for_ts - self.ques_encoding -self.KC_one_hot_dim< self.before_trans_alignment_d:
            for_ts += 64

        self.align_without_onehot_ques = for_ts - self.ques_encoding- self.KC_one_hot_dim
        # 然后再接上一个线性层
        self.after_trans_alignment_d_model = for_ts



        # 这样修改之后，只需在原始代码上加一个concat即可，
        # 也就是说，ques的one-hot一定不能破坏了送进模型




        self.d_bert = 768

        self.ques_trans_epochs = BCQL_EPOCH

        self.num_decoder_layers = BCQL_LAYER
        self.num_encoder_layers = BCQL_LAYER


QP2KT_EPOCH=60
class QP2KT_Config:
    """
     QP2KT

    path interactive attn+ normal attn + question
    先用原生 LSTM
    """

    def __init__(self, sconfig):
        self.config = sconfig
        self.total_attention_weight = 0.15
        # self.before_trans_alignment_d = 1240 + 768
        self.scale_bert = 64
        if self.config.use_all_assignments:
            self.ques_encoding = 100
        else:
            self.ques_encoding = 20
        self.score_encoding = 10
        # self.lr = 0.001
        self.lr = 0.0005
        self.time_encoding = 10

        # transform了一下 不好的话后面改回去
        self.code2vec_dim = 300

        self.code2vec_trans_dim = 256


        self.d_bert = 768
        self.KC_one_hot_dim=20
        # self.kc_embed_dim = 256
        self.kc_embed_dim = 320
        # self.ques_query_64
        # readdata时使用
        self.for_readdata = (self.code2vec_dim + self.ques_encoding +
                             self.d_bert + self.score_encoding + self.time_encoding + self.KC_one_hot_dim)
        # rnn dim
        # 这里的*2 一次是code2vec path concat，一次是fit_trans前concat

        #kc只能一次 self.ques_encoding第一个被trans成256了 所以只有一个
        self.before_trans_alignment_d = (self.kc_embed_dim +  self.ques_encoding +
                        self.scale_bert + self.KC_one_hot_dim+
                                         self.code2vec_dim+self.ques_encoding) # 这是normal attn code vec

        self.rnn_dim=self.before_trans_alignment_d


        self.num_heads = 2

        # 在无self attentiono 的lstm中，不需要关注下面的内容

        # 先用before过一个线性层，这个线性层是最终transformer输入维度（64倍数）-20
        for_ts = ROUNDUP(self.before_trans_alignment_d)
        while for_ts - self.ques_encoding -self.KC_one_hot_dim< self.before_trans_alignment_d:
            for_ts += 64

        self.align_without_onehot_ques = for_ts - self.ques_encoding- self.KC_one_hot_dim
        # 然后再接上一个线性层
        self.after_trans_alignment_d_model = for_ts



        # 这样修改之后，只需在原始代码上加一个concat即可，
        # 也就是说，ques的one-hot一定不能破坏了送进模型




        self.d_bert = 768

        self.ques_trans_epochs = BCQL_EPOCH

        self.num_decoder_layers = BCQL_LAYER
        self.num_encoder_layers = BCQL_LAYER

QPALA_EPOCH=50
QPALA_lr=0.0005
QPALA_attn_lamda=0.1 # 需要去模型具体配置类里修改，同时证明了一点，如果什么都不变，固定随机数会确保结果一致
class QPALA_Config:
    """
     QPALA   path interactive attention + attn after LSTM
    """

    def __init__(self, sconfig):
        self.config = sconfig
        self.total_attention_weight = 0.1
        # self.before_trans_alignment_d = 1240 + 768
        self.scale_bert = 64
        if self.config.use_all_assignments:
            self.ques_encoding = 100
        else:
            self.ques_encoding = 20
        self.score_encoding = 10
        # self.lr = 0.001
        self.lr = QPALA_lr
        self.time_encoding = 10

        # transform了一下 不好的话后面改回去
        self.code2vec_dim = 300

        self.code2vec_trans_dim = 256


        self.d_bert = 768
        self.KC_one_hot_dim=20
        # self.kc_embed_dim = 256
        self.kc_embed_dim = 320
        # self.ques_query_64
        # readdata时使用
        self.for_readdata = (self.code2vec_dim + self.ques_encoding +
                             self.d_bert + self.score_encoding + self.time_encoding + self.KC_one_hot_dim)
        # rnn dim
        # 这里的*2 一次是code2vec path concat，一次是fit_trans前concat

        #kc只能一次 self.ques_encoding第一个被trans成256了 所以只有一个
        self.before_trans_alignment_d = (self.kc_embed_dim +  self.ques_encoding +
                        self.scale_bert + self.KC_one_hot_dim+
                                         self.code2vec_dim+self.ques_encoding) # 这是normal attn code vec

        self.rnn_dim=self.before_trans_alignment_d


        self.num_heads = 2

        # 在无self attentiono 的lstm中，不需要关注下面的内容

        # 先用before过一个线性层，这个线性层是最终transformer输入维度（64倍数）-20
        for_ts = ROUNDUP(self.before_trans_alignment_d)
        while for_ts - self.ques_encoding -self.KC_one_hot_dim< self.before_trans_alignment_d:
            for_ts += 64

        self.align_without_onehot_ques = for_ts - self.ques_encoding- self.KC_one_hot_dim
        # 然后再接上一个线性层
        self.after_trans_alignment_d_model = for_ts

        # 这样修改之后，只需在原始代码上加一个concat即可，
        # 也就是说，ques的one-hot一定不能破坏了送进模型

        self.d_bert = 768

        self.ques_trans_epochs = BCQL_EPOCH

        self.num_decoder_layers = BCQL_LAYER
        self.num_encoder_layers = BCQL_LAYER

class QuestionTransformerConfig:
    def __init__(self):
        self.before_trans_alignment_d = 440
        self.after_trans_alignment_d_model = 640
        self.d_bert = 768

        self.ques_trans_epochs = Q_EPOCH

        self.num_decoder_layers = Q_LAYER
        self.num_encoder_layers = Q_LAYER

    def __str__(self):
        return "&&&&&&&&&&&&&&&&&&&&&&&\n\n我是带Question的配置类\n\n&&&&&&&&&&&&&&&&&&&&&&&"


ques_config = QuestionTransformerConfig()
bert_ques_config = BertBaseQuesTransformerConfig()
raw_config = RawTransformerConfig()
cfg = Config()
config = Config()
lstm_config = LSTMConfig()
BQMOD_config = BertQuesModTransformerConfig()
BCQ_config = BertCompressedQuesTransformerConfig(sconfig=config)
CBTQ_config = CodeBert_BertQuestion_TransformerConfig()
CT5QER_config = CodeT5_BertQuestion_TransformerConfig_Exercise_Response(sconfig=config)
CT5LSTM_config = CT5LSTM(sconfig=config)
CT5Q_config = CodeT5_BertQuestion_TransformerConfig(sconfig=config)
CT5QNT_config = CT5Q_No_time_Config(sconfig=config)

BCQ_LSTMA_config = BertCompressedQuesAttentionLSTMConfig(sconfig=config)
BLAA_config = BLSTM_Attention_After_config(sconfig=config)
KCA_config=KCA_Config(sconfig=config)
KCLSTMA_config=KCLSTMA_Config(sconfig=config)
QPKT_config=QPKT_Config(sconfig=config)
QP2KT_config=QP2KT_Config(sconfig=config)
QPALA_config=QPALA_Config(sconfig=config)



ALL_MODELS = ["QuesTrans", "lstm", "RawTrans", "BERT_QUES_Trans", "BQMOD",
              "BCQ", "CT5Q", "CT5QNT", "CT5QER", "CT5LSTM", "CT5LSTMA",
              "BCQLSTMA", "BCQ_CT", "BCQ_LSTM", "BLAE", "BLAA","KCA","KCLSTMA"
                ,"QPKT","QP2KT","QPALA"]

# ["QuesTrans","lstm","RawTrans"]
model2config = {"QuesTrans": ques_config, "lstm": cfg, "RawTrans": raw_config, "BERT_QUES_Trans": bert_ques_config}
print(f"单例:  :{id(cfg) == id(cfg)}")
print(f"assignment id {Config.ASSIGNMENT_ID}")
