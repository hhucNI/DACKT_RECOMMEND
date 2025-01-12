import math
import random
import torch
import torch.nn as nn
from config import *
import os
from torchsummary import summary
import MyUtils

MAX_CODE_LEN = 100



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


class MyDecoderEmbedding(nn.Module):
    def __init__(self, d_model):
        super(MyDecoderEmbedding, self).__init__()
        self.d_model = d_model
        # self.word_embed_len = word_embed_len

    def forward(self, x):
        # (128,49)
        x = x.view(x.size(0), x.size(1), 1)
        res = x.expand(-1, -1, self.d_model)
        return res
        # res=torch.zeros((x.size(0),x.size(1),self.d_model))


class WhiteBoxTransformer(nn.Module):
    """
        不包括任何positional encoding & embedding
        change : 加入positional encoding 2023-6-4
    """

    def __init__(self, d_model, device, nhead=8, dropout=0.1, num_encoder_layers=4, num_decoder_layers=4):
        super(WhiteBoxTransformer, self).__init__()
        self.device=device

        # d_model=d_model, num_encoder_layers=2, num_decoder_layers=2,
        # dim_feedforward=d_model, batch_first=True
        # encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model, dropout, batch_first=True)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        # decoder_layer = nn.TransformerDecoderLayer(d_model,nhead,dim_feedforward,dropout)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, d_model, dropout, batch_first=True)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.d_model = d_model
        self.nhead = nhead
        self.positional_encoding = PositionalEncoding(d_model)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        # shape check
        if src.size(0) != tgt.size(0):
            raise RuntimeError("USER : the batch number of src and tgt must be equal")
        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("USER : the feature number of src and tgt must be equal to d_model")

        src_mask = torch.triu(torch.ones(src.size(1),src.size(1)), diagonal=1) == 1
        src_mask = src_mask.float().masked_fill(src_mask == True, float('-inf')).masked_fill(src_mask == False,
                                                                                             float(0.0))
        src_mask = src_mask.to(self.device)

        memory = self.encoder(src, mask=src_mask)

        mem_mask = torch.triu(torch.ones(memory.size(1), memory.size(1)), diagonal=1) == 1
        mem_mask = mem_mask.float().masked_fill(mem_mask == True, float('-inf')).masked_fill(mem_mask == False,
                                                                                             float(0.0))
        mem_mask = mem_mask.to(self.device)



        # output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
        #                       tgt_key_padding_mask=tgt_key_padding_mask,
        #                       memory_key_padding_mask=memory_key_padding_mask)
        # fixme tgt在inference时长度与src不匹配的问题可以用key_padding解决吧,或许每次生成一个新mask？

        # tgt=tgt[:,:2,:]
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask,memory_mask=mem_mask)
        # output = self.output_layer(output)
        # return output
        # 输出依然是(128,49,640)，与黑盒无不同，inference取最后一个即可
        return output


class TransformerModel(nn.Module):
    # def __init__(self, d_model=128):
    def __init__(self, d_model, input_dim, hidden_dim, layer_dim, output_dim, node_count, path_count, device):
        super(TransformerModel, self).__init__()
        # TODO 输入可能需要拆开，因为嵌入只嵌入path和start，endnode，不嵌入问题以及答信息
        # 定义词向量，词典数为10。我们不预测两位小数。
        # self.d_model=512 # just for transformer self.length = 50 self.questions = 10
        # self.lr = 0.0005 self.bs = 128 self.hidden = 128 self.layers = 1
        # self.assignment = 439 self.code_path_length = 8 self.code_path_width = 2
        self.num_of_questions = input_dim // 2
        self.embed_dropout = nn.Dropout(0.2)
        # self.embed_nodes = nn.Embedding(node_count + 2, self.config.word_embedding_size)  # adding unk and end
        # self.embed_paths = nn.Embedding(path_count + 2, self.config.word_embedding_size)  # adding unk and end
        self.device = device

        # 定义Transformer
        # Todo demo中d_model参数和dim_feedforward参数不一样，这里为了方便改成了一样，后面再说
        # self.transformer = nn.Transformer(d_model=d_model, num_encoder_layers=2, num_decoder_layers=2,
        #                                   dim_feedforward=d_model, batch_first=True)


        self.transformer = WhiteBoxTransformer(d_model, device, num_encoder_layers=BCQ_config.num_encoder_layers,
                                               num_decoder_layers=BCQ_config.num_decoder_layers)


        self.tgt_embed = MyDecoderEmbedding(BCQ_config.after_trans_alignment_d_model)
        # nn.TransformerEncoder()
        # fixme 先expand一下跑起来
        # 定义位置编码器
        self.positional_encoding = PositionalEncoding(d_model, dropout=0)
        self.input_dim = input_dim
        # 定义最后的线性层，这里并没有用Softmax，因为没必要。
        # 因为后面的CrossEntropyLoss中自带了
        self.predictor = nn.Linear(d_model
                                   , output_dim)
        self.sig = nn.Sigmoid()

    # tgt由x生成，类似loss_func的处理
    def forward(self, src, tgt, evaluating=False):
        """"# 0 or 1
        # 对tgt进行编码
        # tgt (128,49)
        # tgt = (((src[:, :, 0:self.num_of_questions] -
        #          src[:, :, self.num_of_questions:self.num_of_questions * 2]).sum(2) + 1) //
        #        2)[:, 1:]
        # tgt=tgt[:,:10]
        # embed_tgt = self.tgt_embed(tgt).to(self.device)
        # embed_tgt = embed_tgt.view(embed_tgt.size(0), embed_tgt.size(1), embed_tgt.size(2) * embed_tgt.size(3))
        # tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1]).to(self.device)"""
        if len(tgt.size()) == 2:
            # expand
            tgt = self.tgt_embed(tgt)

        # tgt = MyUtils.add_BOS_to_transformer_input(tgt, self.device)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[1]).to(self.device)
        tout = self.transformer(src, tgt,
                                tgt_mask=tgt_mask)
        # tout = self.transformer(src, embed_tgt,
        #                         tgt_mask=tgt_mask)

        out = self.sig(self.predictor(tout))
        """
        这里直接返回transformer的结果。因为训练和推理时的行为不一样，
        所以在该模型外再进行线性层的预测。
        """
        return out

    @staticmethod
    def get_key_padding_mask(tokens):
        # tokens=src
        # (128,49,320)

        """
        用于key_padding_mask
        # TODO 检查直接复制的padding mask
        """

        key_padding_mask = torch.zeros(tokens.size())  # (128,49,320)
        # TODO *****这里显然需要大改，如果是2的话会导致很多问题，可能要改特殊标记比如<pad>的对应值
        # 好像也可以不mask
        key_padding_mask[tokens == 2] = -torch.inf

        return key_padding_mask


if __name__ == "__main__":
    node_count = 2222
    path_count = 3333
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = TransformerModel(config.d_model_no_ques,
                             config.questions * 2,
                             config.hidden,
                             config.layers,
                             config.questions,
                             node_count, path_count, device)
    print(model)
    # summary(model=model, input_size=(1,node_count+2,), batch_size=1024, device="cpu")
