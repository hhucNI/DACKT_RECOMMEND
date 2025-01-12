# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import MyUtils
import numpy as np
from src.KCA_InnerRawCoreTransformer import *
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


class BCQ_Attention_LSTM(nn.Module):
    # input : (128,50,320)->(batch_size,max_step,2*question_num+path_nodes_len)
    def __init__(self, d_model, input_dim, hidden_dim, layer_dim, output_dim, node_count, path_count, device):
        super(BCQ_Attention_LSTM, self).__init__()

        self.embed_nodes = nn.Embedding(node_count + 2, 100)  # adding unk and end
        self.embed_paths = nn.Embedding(path_count + 2, 100)  # adding unk and end
        self.embed_dropout = nn.Dropout(0.2)
        self.bertvec_scaling = nn.Linear(BCQ_config.d_bert, BCQ_config.scale_bert)
        self.BERT_ques_dim = 768
        self.path_transformation_layer = nn.Linear(input_dim + 300, input_dim + 300)
        self.attention_layer = nn.Linear(input_dim + 300, 1)
        #         self.feature_layer = nn.Linear(300,10)
        self.prediction_layer = nn.Linear(input_dim + 300, 1)
        self.attention_softmax = nn.Softmax(dim=1)
        self.last_attention_softmax = nn.Softmax(dim=1)
        self.total_attention_softmax = nn.Softmax(dim=1)

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
        # self.fit_transfromer_linear_layer = nn.Linear(BCQ_config.before_trans_alignment_d,
        #                                               BCQ_config.align_without_onehot_ques)
        # TODO 看看这个干嘛的
        # self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        self.sig = nn.Sigmoid()
        self.device = device

        #融合一下特则，最后加上onehot
        self.feature_fusion=nn.Linear(KCA_config.rnn_dim,KCA_config.align_without_onehot_ques)
        self.rnn = nn.LSTM(KCA_config.after_trans_alignment_d_model,
                           self.hidden_dim,
                           1,
                           batch_first=True)
        # self.fit_self_attention_layer=nn.Linear(KCA_config.rnn_dim,KCA_config.align_without_onehot_ques)
        # self.positional_encoding = PositionalEncoding(KCA_config.after_trans_alignment_d_model)

        # self-design Attention

        # 关注out，即所有特征
        self.time_seq_attention_layer=nn.Linear(self.hidden_dim,1)

        # 关注bert ques，仅问题特征
        self.time_seq_attention_layer_focus_on_ques = nn.Linear(KCA_config.scale_bert+KCA_config.KC_one_hot_dim, 1)




        # self.masked_attention = MaskedAttention(KCA_config.after_trans_alignment_d_model, KCA_config.num_heads)
        # 无embedding模块的Transformer 处理融合,处理后的特征
        # callsite that matters
        # self.transformer_model = TransformerModel(CT5LSTM_config.after_trans_alignment_d_model,
        #                                           config.questions,
        #                                           device)

        # OUTPUT do predictions
        # self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.fc = nn.Linear(self.hidden_dim+KCA_config.KC_one_hot_dim+KCA_config.ques_encoding, config.questions)
        self.bert_questions = torch.tensor(np.load(cfg.questions_BERT_embedding_file_name), device=device)






    # TODO 这个question是啥
    def forward(self, src,evaluating=False):
        # shape of input: [batch_size, length=50, questions * 2+c2vnodes]
        #         h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)
        # shape: [num_layers * num_directions, batch_size, hidden_size]
        # TODO 注意这里x(128,49,320)，是删去尾巴的batch_new,
        # 2023-9-22  src (64,50,320)

        x = src.to(self.device)  # x (64,49,320)
        rnn_first_part = x[:, :, :self.input_dim]  # (64,49,20)


        # 问题，时间，ac rate
        ques_bert_id = x[:, :, KCA_config.ques_encoding + KCA_config.code2vec_dim
                               :KCA_config.ques_encoding + KCA_config.code2vec_dim+ KCA_config.d_bert]
        time_id = x[:, :, KCA_config.ques_encoding + KCA_config.code2vec_dim+ KCA_config.d_bert
                          :KCA_config.ques_encoding + KCA_config.code2vec_dim+ KCA_config.d_bert+KCA_config.time_encoding]
        score_id = x[:, :, KCA_config.ques_encoding + KCA_config.code2vec_dim+ KCA_config.d_bert+KCA_config.time_encoding
                           :KCA_config.for_readdata-KCA_config.KC_one_hot_dim]
        kc_part=x[:,:,KCA_config.for_readdata-KCA_config.KC_one_hot_dim:]


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
        # 这里释放显存

        # starting_node_embed=None
        # ending_node_embed=None
        # path_embed=None
        # torch.cuda.empty_cache()
        # path_transformation_layer这个不知道干嘛用的，形状不变 trans (64,49,100,320)
        trans = self.path_transformation_layer(full_embed)
        full_embed_transformed = torch.tanh(trans)
        context_weights = self.attention_layer(full_embed_transformed)  # (64,49,100,1)
        attention_weights = self.attention_softmax(context_weights)  # (64,49,100,1)
        code_vectors = torch.sum(torch.mul(full_embed, attention_weights), dim=2)

        #  (64,49,20+320=340)
        # change (BS,49,20+1220+768)
        bert_vec = self.bertvec_scaling(ques_bert_id)
        rnn_input = torch.cat((rnn_first_part, code_vectors, bert_vec), dim=2)

        # trans_input (64,49,640)
        # print(f"KCA_config.rnn_dim {KCA_config.rnn_dim}")
        rnn_full_input = torch.cat((time_id,score_id, rnn_input,kc_part), dim=2)
        fusioned_rnn_input=self.feature_fusion(rnn_full_input)
        fusioned_rnn_input=torch.cat((kc_part,rnn_first_part, fusioned_rnn_input), dim=2)

        out = self.rnn(fusioned_rnn_input)
        # redesign output
        attended_out=torch.zeros(size=(out[0].shape[0],out[0].shape[1],self.hidden_dim)).to(self.device)
        attended_out=out[0]*(1-KCA_config.total_attention_weight) #所有时间步置为 1-λ
        # attended_out=out[0] #所有时间步置为 1
        attended_out[:,0,:]=out[0][:,0,:]  #第一个时间步无法attn，故为1

        # ------------------------------------------------------------------------
        # attn关注out，也就是所有特征
        # ------------------------------------------------------------------------

        # for i in range(1,cfg.length-1):
        #     partial_out=out[0][:,:i,:]
        #     per_time_step_attn_weight=self.time_seq_attention_layer(partial_out)
        #     last_per_time_step_attn_weight=self.last_attention_softmax(per_time_step_attn_weight)
        #     cur_step_out=torch.sum(torch.mul(partial_out, last_per_time_step_attn_weight), dim=1)
        #     attended_out[:,i,:]+=cur_step_out*KCA_config.total_attention_weight
        # before_norm=self.fc(attended_out)
        # res = self.sig(before_norm)  # shape of res: [batch_size, length, question]
        # return res


        # ------------------------------------------------------------------------
        # attn关注之前的问题特征，感觉会更好  更新是在attend out后面又加了onehot
        # 这个有时间衰减
        # ------------------------------------------------------------------------
        for i in range(1, cfg.length - 1):
            partial_ques_feature = torch.cat((bert_vec[:, :i + 1, :], kc_part[:, :i + 1, :]), dim=2)
            partial_out = out[0][:, :i + 1, :]
            per_time_step_attn_weight = self.time_seq_attention_layer_focus_on_ques(partial_ques_feature)
            last_per_time_step_attn_weight = self.last_attention_softmax(per_time_step_attn_weight)
            cur_step_out = torch.sum(torch.mul(partial_out, last_per_time_step_attn_weight), dim=1)
            attended_out[:, i, :] += cur_step_out * BLAA_config.total_attention_weight
        add_encode_attended_out = torch.cat((rnn_first_part,kc_part, attended_out), dim=2)
        before_norm = self.fc(add_encode_attended_out)
        res = self.sig(before_norm)  # shape of res: [batch_size, length, question]
        return res






if __name__=="__main__":
    # Test with random input
    input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN,
                               EMBED_DIM)  # Batch size: 32, Sequence length: 49, Embedding dimension: 1000
    masked_self_attention = MaskedSelfAttention(num_heads=NUM_HEADS)
    output_tensor = masked_self_attention(input_tensor)
    print(output_tensor.shape)  # Expected output: torch.Size([32, 49, 1000])
