# 492
# 494
# 439
# 502
# 487

import preprocessing
import torch
import path_extractor
# import get_ques_word_idx
from compare_lstm_and_trans import *
import time
import BQ_get_question_embedding as BertQuesEmbedding

# import RT_runRawTrans as RawTransformer
# import QT_run as QuesTransformer
# import CT5Q_runcodeBertRawTrans as CT5Q
# import BQ_runRawTrans as BQ_Ques_Transformer
# import BQMOD_runRawTrans as BQ_MOD_runner
# import CT5LSTM_runcodeBertRawTrans as CT5LSTM_runner
# import CT5LSTMA_runcodeBertRawTrans as CT5LSTMA_runner
# import CT5QER_runcodeBertRawTrans as CT5QER_runner
# import BCQLSTMA_runRawTrans as BCQLSTMA_runner
# import BCQLSTM_runRawTrans as BCQLSTM_runner
#
# import CT5QNotime_runcodeBertRawTrans as CT5QNT
# import BCQT_runRawTrans as BCQ_CT_runner
# import KCLSTMA_runRawTrans as KCLSTMA_runner
# import QPKT_runRawTrans as QPKT_runner
# import QP2KT_runRawTrans as QP2ATTNKT_runner
# import KCA_runRawTrans as KCA_runner
# import BLAA_runRawTrans as BLAA_runner

import runOriginal as LSTM
import QPALA_runRawTrans as QPALA_runner

# assignments={439:False,492:False,494:False,502:False,487:False}
# assignments={439,492,494,502,487]
# for a in assignments:



# from config import *
start = time.time()

# 目前结构下做不到遍历assignment，因为import时就已经确定
# # with open(os.path.join(cfg.comparison_root, "README.txt"), "a") as f:
# #     text = f"{Config.ASSIGNMENT_ID} :Done\n"
# #     f.write(text)
# # print(f"DEVICE :    {torch.cuda.get_device_name(0)}")
# # #


use_all_assignments = False
preprocessing.main(use_all_assignments=use_all_assignments)
path_extractor.main(use_all_assignments=use_all_assignments)
#
# # get_ques_word_idx.main()
#
# #'
BertQuesEmbedding.main()
print("预处理完毕")
print("模型验证2")


# raise
#
# #
# # CT5QNT.main()

# print("----------------------------------BCQ-------BCQ---------------------------")
# BCQ_runner.main()
# BCQ_CT_runner.main()
# BQ_Ques_Transformer.main()
# raise



# print("----------------------------------KCA ---------------------------")
# KCA_runner.main()

# print("----------------------------------KC LSTM A ---------------------------")
# KCLSTMA_runner.main()

# print("----------------------------------BCQ LSTMA---------------------------")
# BCQLSTMA_runner.main()
# print("----------------------------------BLAA ---------------------------")
# BLAA_runner.main()
# print("----------------------------------BCQ raw LSTM ---------------------------")
# BCQLSTM_runner.main()


# print("-----------------------------   QP2 ATTN KT   ---------------------------")
# QP2ATTNKT_runner.main()


print("-----------------------------   QPALA  ---------------------------")
QPALA_runner.main()

# print("-----------------------------   QPKT   ---------------------------")
# QPKT_runner.main()

print("----------------------------------LSTM---------------------------")
LSTM.main()

# print("----------------------------------CT5 ER---------------------------")
#
# CT5QER_runner.main()
# CT5LSTMA_runner.main()
# print("----------------------------------CT5Q---------------------------")
#
# CT5Q.main()






#
#
# RawTransformer.main()


# deprecated
# QuesTransformer.main()

#
print("效果评估")

# "QuesTrans"  "lstm"  "RawTrans" "BERT_QUES_Trans"

# res = get_json_info_fromNfiles(["QuesTrans","lstm","RawTrans","BERT_QUES_Trans"])





res = get_json_info_fromNfiles(config.model_to_run)



# res = get_json_info_fromNfiles(["lstm","KCLSTMA"])
# res = get_json_info_fromNfiles(["lstm","CT5LSTM"])
# res = get_json_info_fromNfiles(["lstm", "RawTrans", "BCQ", "CT5Q"])

# trans_info, lstm_info = get_json_info_from2files()
# ques_info=res["QuesTrans"]
# lstm_info=res["lstm"]
# raw_info=res["RawTrans"]
# for fold in range(config.fold):
#     compare2model_per_fold(fold, trans_info, lstm_info)


#TORUN
compareNmodel_avg_totalvg_total(res, "first_avg")
compareNmodel_avg_total(res, "all_avg")



# compare2model_avg_total(trans_info, lstm_info, "first_avg")
# compare2model_avg_total(trans_info, lstm_info, "all_avg")

# generate_model_results_table()
score_tables_for_all_ass()


end = time.time()
print("-----------------------------------------------------------")
print(f"---------         总耗时 {end - start} s         ------------")
print("-----------------------------------------------------------")
