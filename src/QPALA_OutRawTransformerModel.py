# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import MyUtils
import numpy as np
from src.config import *

MAX_CODE_LEN = 100

import torch
import torch.nn as nn
import torch.nn.functional as F




# Define global constants
BATCH_SIZE = 32
SEQ_LEN = 49
EMBED_DIM = 1000
NUM_HEADS = 8
FOCUS_DIM = 500  # Number of features to focus on, in this case, the first 500 features




class PositionalEncoding(nn.Module):
    "Implement the PE function."

    """
    此处d_model即transformer维度，为simple_d_model=640
    """

    def __init__(self, d_model, dropout=0.1, max_len=50, enable_dropout=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.enable_dropout = enable_dropout
        # (length,320还是300,d_model)
        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        # 10000^-(i/d_model)
        # 0,2,4,......,d_model

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        if self.enable_dropout:
            return self.dropout(x)
        return x








class KC_Attention_LSTM(nn.Module):
    # input : (128,50,320)->(batch_size,max_step,2*question_num+path_nodes_len)
    def __init__(self, d_model, input_dim, hidden_dim, layer_dim, output_dim, node_count, path_count, device):
        super(KC_Attention_LSTM, self).__init__()

        self.embed_nodes = nn.Embedding(node_count + 2, 100)  # adding unk and end
        self.embed_paths = nn.Embedding(path_count + 2, 100)  # adding unk and end
        self.embed_dropout = nn.Dropout(0.2)
        self.bertvec_scaling = nn.Linear(QPALA_config.d_bert, QPALA_config.scale_bert)
        self.BERT_ques_dim = 768
        # self.path_transformation_layer = nn.Linear(input_dim + 300, input_dim + 300)
        self.ia_path_transformation_layer = nn.Linear(input_dim + 300, QPALA_config.kc_embed_dim)
        self.na_path_transformation_layer = nn.Linear(input_dim + 300,input_dim + 300 )
        self.attention_layer = nn.Linear(input_dim + 300, 1)
        #         self.feature_layer = nn.Linear(300,10)
        self.prediction_layer = nn.Linear(input_dim + 300, 1)
        self.attention_softmax = nn.Softmax(dim=1)

        self.input_dim = input_dim
        self.num_of_questions = input_dim // 2
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        # lstm

        # TODO transformer加入dropout试试
        # self.dropout = nn.Dropout(p=0.1)

        # TODO 看看这个干嘛的
        # self.sig = nn.Sigmoid()

        self.device = device
        # self.fit_transfromer_linear_layer = nn.Linear(QPALA_config.before_trans_alignment_d,
        #                                               QPALA_config.align_without_onehot_ques)
        # TODO 看看这个干嘛的
        # self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        self.sig = nn.Sigmoid()
        self.device = device

        # 关注bert ques，仅问题特征
        self.time_seq_attention_layer_focus_on_ques = nn.Linear(QPALA_config.scale_bert + QPALA_config.KC_one_hot_dim+QPALA_config.kc_embed_dim, 1)
        self.last_attention_softmax = nn.Softmax(dim=1)
        self.total_attention_softmax = nn.Softmax(dim=1)
        # OUTPUT do predictions
        # self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.fc = nn.Linear(self.hidden_dim+QPALA_config.KC_one_hot_dim+QPALA_config.ques_encoding, config.questions)
        self.bert_questions = torch.tensor(np.load(cfg.questions_BERT_embedding_file_name), device=device)

        # 假设词汇表大小为10000，每个单词的嵌入向量维度为256
        self.kc_embedding_layer = nn.Embedding(num_embeddings=QPALA_config.KC_one_hot_dim, embedding_dim=QPALA_config.kc_embed_dim)
        
        
        self.kc_avg_embedding=KC_Avg_Embedding(QPALA_config.KC_one_hot_dim,QPALA_config.kc_embed_dim)
        
        
        self.rnn = nn.LSTM(QPALA_config.rnn_dim,
                           self.hidden_dim,
                           1,
                           batch_first=True)

        # KCA_config.KC_one_hot_dim >= concept+1
        self.path_attn_kc_embedding=self.concept_embedding = nn.Parameter(torch.randn(QPALA_config.KC_one_hot_dim, QPALA_config.kc_embed_dim), requires_grad=True)

    # TODO 这个question是啥
    def forward(self, src,evaluating=False):
        # shape of input: [batch_size, length=50, questions * 2+c2vnodes]
        #         h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)
        # shape: [num_layers * num_directions, batch_size, hidden_size]
        # TODO 注意这里x(128,49,320)，是删去尾巴的batch_new,
        # 2023-9-22  src (64,50,320)

        x = src.to(self.device)  # x (64,49,320)
        rnn_first_part = x[:, :, :self.input_dim]  # (64,49,20)


        # 问题，时间，ac rate,kc
        ques_bert_id = x[:, :, QPALA_config.ques_encoding + QPALA_config.code2vec_dim
                               :QPALA_config.ques_encoding + QPALA_config.code2vec_dim + QPALA_config.d_bert]
        time_id = x[:, :, QPALA_config.ques_encoding + QPALA_config.code2vec_dim + QPALA_config.d_bert
                          :QPALA_config.ques_encoding + QPALA_config.code2vec_dim + QPALA_config.d_bert + QPALA_config.time_encoding]
        score_id = x[:, :,
                   QPALA_config.ques_encoding + QPALA_config.code2vec_dim + QPALA_config.d_bert + QPALA_config.time_encoding
                   :QPALA_config.for_readdata - QPALA_config.KC_one_hot_dim]


        kc_part = x[:, :, QPALA_config.for_readdata - QPALA_config.KC_one_hot_dim:]


        # indices_tensor = torch.tensor([0, 0, 1, 0, 2, 3, 0, 0, 4, 5, 0, 6, 0, 0, 7, 8, 0, 9, 0, 10], dtype=torch.long)

        # 找到非零索引（即值为1的索引，但这里我们假设所有非零值都表示我们关心的索引）
        # 注意：如果张量中只有0和1，且1表示有效索引，可以直接使用indices_tensor（如果它是long类型）
        # 但如果索引不是连续的，你可能需要筛选非零索引

        # 我现在有个20维的向量，有些位置为1，有些位置为0，我希望把它扩展成20 * 20
        # 的矩阵，每一行是一个onehot向量或者全0向量，
        # 根据该位置是否为1决定，然后通过一个嵌入层得到每行的嵌入，
        # 然后对每行的嵌入做sum然后根据原始向量为1的位置数量做平均值


        # avg_kc_embed_part=torch.zeros(kc_part.)
        # for kc_row in kc_part:
        #     kc_non_zero_indices = kc_row.nonzero(as_tuple=False).squeeze()  # 如果indices_tensor中有多个非零值
        # 
        #     # 使用非零索引从嵌入层中获取嵌入向量
        #     # 注意：如果indices_tensor中只有0和1，并且你的目的是简单地使用1作为有效索引，你可以直接使用indices_tensor（如果它是long且没有多余的0）
        #     kc_embeddings =self.kc_embedding_layer(kc_non_zero_indices)
        #     kc_num_per_row=kc_embeddings.shape[-2]
        #     avg_kc_row=kc_embeddings.sum(dim=-2)/kc_num_per_row
            
            
            
        #应该使用embedding和代码作interaction来得到本题的表示
        #先跑一个MVP 直接concat知识点平均值嵌入
        # kc_nums=torch.sum(kc_part,dim=2)
        
        
        
        
        
        # 封装好的KC_Avg_Embedding
        # kc_embed=self.kc_avg_embedding(kc_part)
        # concat 到结果
        
        

        rnn_attention_part = torch.stack([rnn_first_part] * MAX_CODE_LEN, dim=-2)  # (b,l,c,2q)
        # rnn_attention_part (64,49,100,20)
        c2v_input = x[:, :, config.questions * 2:config.questions * 2 + MAX_CODE_LEN * 3].reshape(x.size(0), x.size(1),
                                                                                                  MAX_CODE_LEN,
                                                                                                  3).long()  # (b,l,c,3)(b,l,100,3)
        # c2v_input (64,49,100,3)
        starting_node_index = c2v_input[:, :, :, 0]
        ending_node_index = c2v_input[:, :, :, 2]  # (b,l,c,1)
        path_index = c2v_input[:, :, :, 1]
        # starting_node_index (64,49,100)

        starting_node_embed = self.embed_nodes(starting_node_index)  # (b,l,c,1) -> (b,l,c,ne)
        ending_node_embed = self.embed_nodes(ending_node_index)  # (b,l,c,1) -> (b,l,c,ne)
        path_embed = self.embed_paths(path_index)  # (b,l,c,1) -> (b,l,c,pe)
        # starting_node_embed (64,49,100,100)

        full_embed = torch.cat((starting_node_embed, ending_node_embed, path_embed, rnn_attention_part),
                               dim=3)  # (b,l,c,2ne+pe+q) # full_embed (64,49,100,320)
        if not evaluating:
            full_embed = self.embed_dropout(full_embed)  # (b,l,c,2ne+pe+2q)

        bert_vec = self.bertvec_scaling(ques_bert_id)


        # -----------------------------------------------------------------------
        #                                  normal attention
        # -----------------------------------------------------------------------

        trans = self.na_path_transformation_layer(full_embed)
        full_embed_transformed = torch.tanh(trans)
        context_weights = self.attention_layer(full_embed_transformed)  # (64,49,100,1)
        attention_weights = self.attention_softmax(context_weights)  # (64,49,100,1)
        na_code_vectors = torch.sum(torch.mul(full_embed, attention_weights), dim=2)
        # na_code_vectors (64,49,320)

        # -----------------------------------------------------------------------
        #                              interactive attention
        # -----------------------------------------------------------------------

        kc_num = torch.sum(kc_part, dim=2).unsqueeze(2)
        kc_num_no_zero = torch.masked_fill(kc_num, kc_num.eq(0), 1)
        # 本模型中 out-dim为 256
        trans = self.ia_path_transformation_layer(full_embed)

        bs = kc_part.shape[0]
        seqlen = kc_part.shape[1]

        expanded_kc_part = kc_part.unsqueeze(3).expand(bs, seqlen, QPALA_config.KC_one_hot_dim,
                                                       QPALA_config.kc_embed_dim)
        expanded_kc_embedding = self.path_attn_kc_embedding.repeat(bs, seqlen, 1, 1)
        Q = expanded_kc_part * expanded_kc_embedding
        Q_trans = Q.view(bs*seqlen, QPALA_config.KC_one_hot_dim, QPALA_config.kc_embed_dim)
        # path 信息
        Vt_N_Kt_trans = trans.view(bs * seqlen, MAX_CODE_LEN, QPALA_config.kc_embed_dim)
        QK_T = torch.bmm(Q_trans,
                                    Vt_N_Kt_trans.permute(0, 2, 1))  # 1600,38,200 N,L

        softmax_QK_T = torch.softmax(QK_T, dim=2)
        H = torch.bmm(softmax_QK_T, Vt_N_Kt_trans)
        H_TM=H*expanded_kc_part.view(bs * seqlen, QPALA_config.KC_one_hot_dim, QPALA_config.kc_embed_dim)
        ia_code_vectors = torch.sum(H_TM, dim=1) / kc_num_no_zero.view(-1).unsqueeze(1)
        ia_code_vectors=ia_code_vectors.view(bs, seqlen, QPALA_config.kc_embed_dim)
        # attn_before_full_input = torch.cat((kc_part,time_id,score_id,rnn_first_part, code_vectors, bert_vec,kc_embed), dim=2)



        rnn_input = torch.cat((kc_part,rnn_first_part, ia_code_vectors,na_code_vectors, bert_vec), dim=2)
        # feature fusion 这里要不要再做一次
        lstm_out, (hidden, cell) = self.rnn(rnn_input)

        # ------------------------------------------------------------------------
        #                                 attn初始化
        # ------------------------------------------------------------------------

        attended_out = torch.zeros(size=(lstm_out.shape[0], lstm_out.shape[1], self.hidden_dim)).to(self.device)
        attended_out = lstm_out * (1 - QPALA_config.total_attention_weight)  # 所有时间步置为 1-λ
        # attended_out=out[0] #所有时间步置为 1
        attended_out[:, 0, :] = lstm_out[:, 0, :]  # 第一个时间步无法attn，故为1
        # ------------------------------------------------------------------------
        # attn关注之前的问题特征，感觉会更好  更新是在attend out后面又加了onehot
        # 这个有时间衰减
        # ------------------------------------------------------------------------
        for i in range(1, cfg.length - 1):
            partial_ques_feature = torch.cat((bert_vec[:, :i + 1, :], kc_part[:, :i + 1, :],ia_code_vectors[:, :i + 1, :]), dim=2)
            partial_out = lstm_out[:, :i + 1, :]
            per_time_step_attn_weight = self.time_seq_attention_layer_focus_on_ques(partial_ques_feature)
            last_per_time_step_attn_weight = self.last_attention_softmax(per_time_step_attn_weight)
            cur_step_out = torch.sum(torch.mul(partial_out, last_per_time_step_attn_weight), dim=1)
            attended_out[:, i, :] += cur_step_out * BLAA_config.total_attention_weight
        add_encode_attended_out = torch.cat((rnn_first_part, kc_part, attended_out), dim=2)
        before_norm = self.fc(add_encode_attended_out)
        res = self.sig(before_norm)  # shape of res: [batch_size, length, question]
        return res


class KC_Avg_Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(KC_Avg_Embedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_matrix = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
    def forward(self, batch_vectors):
        # batch_vectors: (batch_size, seq_length, 20)

        batch_size, seq_length, vec_dim = batch_vectors.size()

        # 扩展成20x20矩阵
        onehot_matrices = torch.zeros((batch_size, seq_length, vec_dim, vec_dim)).to(batch_vectors.device)
        for i in range(vec_dim):
            onehot_matrices[:, :, i, i] = batch_vectors[:, :, i]

        # 通过嵌入层得到每行的嵌入
        # embedded = self.embedding_matrix[onehot_matrices.long()]
        embedded = onehot_matrices @ self.embedding_matrix

        # 对每行的嵌入做sum
        sum_embeddings = embedded.sum(dim=2)  # (batch_size, seq_length, vec_dim, embedding_dim)

        # 计算平均值
        num_ones = batch_vectors.sum(dim=2, keepdim=True).clamp(min=1)  # (batch_size, seq_length, 1)
        avg_embeddings = sum_embeddings / num_ones  # (batch_size, seq_length, embedding_dim)

        return avg_embeddings

if __name__=="__main__":
    # Test with random input
    input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN,
                               EMBED_DIM)  # Batch size: 32, Sequence length: 49, Embedding dimension: 1000
    # masked_self_attention = MaskedSelfAttention(num_heads=NUM_HEADS)
    # output_tensor = masked_self_attention(input_tensor)
    # print(output_tensor.shape)  # Expected output: torch.Size([32, 49, 1000])
