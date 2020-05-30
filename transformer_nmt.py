# 首先来说一下Transformer和LSTM的最大区别，就是LSTM的训练是迭代（自回归）的，是一个接一个字的来，
# 当前这个字过完LSTM单元，才可以进下一个字，而 TransformerTransformer 的训练是并行了，就是所有字是全部同时训练的，
# 这样就大大加快了计算效率，TransformerTransformer 使用了位置嵌入(positional encoding)(positional encoding)来理解语言的顺序，
# 使用自注意力机制和全连接层来进行计算，这些后面都会详细讲解。
#
# TransformerTransformer 模型主要分为两大部分，分别是编码器（EncoderEncoder）和解码器（DecoderDecoder）：
#
#     编码器（EncoderEncoder）负责把自然语言序列映射成为隐藏层(下图中第2步用九宫格比喻的部分)，含有自然语言序列的数学表达
#     解码器（DecoderDecoder）再把隐藏层映射为自然语言序列，从而使我们可以解决各种问题，
#     如情感分类、命名实体识别、语义关系抽取、摘要生成、机器翻译等等。

import os
import math
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nltk import word_tokenize
from collections import Counter
from torch.autograd import Variable

# 初始化参数设置
UNK = 0  # 未登录词的标识符对应的词典id
PAD = 1  # padding占位符对应的词典id
BATCH_SIZE = 64  # 每批次训练数据数量
# EPOCHS = 20  # 训练轮数
EPOCHS = 20  # 训练轮数
LAYERS = 6  # transformer中堆叠的encoder和decoder block层数
H_NUM = 8  # multihead attention hidden个数
D_MODEL = 256  # embedding维数
D_FF = 1024  # feed forward第一个全连接层维数
DROPOUT = 0.1  # dropout比例
MAX_LENGTH = 60  # 最大句子长度

TRAIN_FILE = 'nmt/en-cn/train.txt'  # 训练集数据文件
DEV_FILE = "nmt/en-cn/dev.txt"  # 验证(开发)集数据文件
SAVE_FILE = 'save/model.pt'  # 模型保存路径(注意如当前目录无save文件夹需要自己创建)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
对一个batch批次(以单词id表示)的数据进行padding填充对齐长度
获取批次中的最大句子长度
小于此长度的用 np.concatenate([x, [padding] * (ML - len(x))]) 补齐
"""
def seq_padding(X, padding=0):
    """
    对一个batch批次(以单词id表示)的数据进行padding填充对齐长度
    """
    # 计算该批次数据各条数据句子长度
    L = [len(x) for x in X]
    # 获取该批次数据最大句子长度
    ML = max(L)
    # 对X中各条数据x进行遍历，如果长度短于该批次数据最大长度ML，则以padding id填充缺失长度ML-len(x)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class PrepareData:
    def __init__(self, train_file, dev_file):

        """
        读取数据 并分词
        'nmt/en-cn/train.txt'  # 训练集数据文件
        'nmt/en-cn/dev.txt"'   # 验证(开发)集数据文件
        en = [['BOS', 'i', 'love', 'you', 'EOS'], ['BOS', 'me', 'too', 'EOS'], ...]
        cn = [['BOS', '我', '爱', '你', 'EOS'], ['BOS', '我', '也', '是', 'EOS'], ...]
        """

        self.train_en, self.train_cn = self.load_data(train_file)
        self.dev_en, self.dev_cn = self.load_data(dev_file)

        """
        构建单词表
        word_dict: word(key):index(id)
        word_dict["UNK"] = 0; //未登陆词
        word_dict["PAD"] = 1; //占位符
        total_words: 语料库总的单词数量(包含UNK和PAD)
        index_dict: index(key):word(id)
        """
        self.en_word_dict, self.en_total_words, self.en_index_dict = self.build_dict(self.train_en)
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = self.build_dict(self.train_cn)

        """
        id化
        这里使用了一个Trick,做批次化的时候可以减少padding量
        如果sort参数设置为True，则会以翻译前(英文)的句子(单词数)长度排序
        以便后续分batch做padding时，同批次各句子需要padding的长度相近减少padding量
        en句子列表按长度从小到大排序
        """
        self.train_en, self.train_cn = self.wordToID(self.train_en, self.train_cn, self.en_word_dict, self.cn_word_dict)
        self.dev_en, self.dev_cn = self.wordToID(self.dev_en, self.dev_cn, self.en_word_dict, self.cn_word_dict)

        """
        划分batch + padding + mask
        """
        self.train_data = self.splitBatch(self.train_en, self.train_cn, BATCH_SIZE)
        self.dev_data = self.splitBatch(self.dev_en, self.dev_cn, BATCH_SIZE)

    def load_data(self, path):
        """
        读取翻译前(英文)和翻译后(中文)的数据文件

        I'm opposed to any type of war.	我反對任何形式的戰爭。
        I had a hard day.	我过了难挨的一天。
        He will succeed to the throne.	他会继承王位。
        Please give me something hot to drink.	請給我一些熱的東西喝。
        Have fun.	玩得開心。


        每条数据都进行分词，然后构建成包含起始符(BOS)和终止符(EOS)的单词(中文为字符)列表
        形式如：en = [['BOS', 'i', 'love', 'you', 'EOS'], ['BOS', 'me', 'too', 'EOS'], ...]
                cn = [['BOS', '我', '爱', '你', 'EOS'], ['BOS', '我', '也', '是', 'EOS'], ...]
        """
        en = []
        cn = []
        # TODO ...
        with open(path, 'r', encoding='utf-8') as fin:
            for line in fin:
                list_content = line.split('\t')
                # print(list_content)
                en.append(['BOS'] + word_tokenize(list_content[0]) + ['EOS'])
                # test_1 = word_tokenize(list_content[1])
                cn.append(['BOS'] + word_tokenize(" ".join(list_content[1])) + ['EOS'])

        # print(cn[:10])
        return en, cn

    def build_dict(self, sentences, max_words=50000):
        """
        传入load_data构造的分词后的列表数据
        构建词典(key为单词，value为id值)
        获取前max_words个单词
        """
        # 对数据中所有单词进行计数
        word_count = Counter()

        for sentence in sentences:
            for s in sentence:
                word_count[s] += 1

        """    
        只保留最高频的前max_words数的单词构建词典
        并添加上UNK和PAD两个单词，对应id已经初始化设置过
        UNK = 0  # 未登录词的标识符对应的词典id
        PAD = 1  # padding占位符对应的词典id
        """

        ls = word_count.most_common(max_words)
        # 统计词典的总词数
        total_words = len(ls) + 2

        # for index, w in enumerate(ls):
        #     if index > 10:
        #         continue
        #     print(index, w)

        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
        word_dict['UNK'] = UNK
        word_dict['PAD'] = PAD

        """
        再构建一个反向的词典，供id转单词使用
        id(key) : word(value)
        """
        index_dict = {v: k for k, v in word_dict.items()}

        return word_dict, total_words, index_dict

    def wordToID(self, en, cn, en_dict, cn_dict, sort=True):
        """
        该方法可以将翻译前(英文)数据和翻译后(中文)数据的单词列表表示的数据
        均转为id列表表示的数据
        如果sort参数设置为True，则会以翻译前(英文)的句子(单词数)长度排序
        以便后续分batch做padding时，同批次各句子需要padding的长度相近减少padding量
        """
        # 计算英文数据条数
        length = len(en)

        """
        将翻译前(英文)数据和翻译后(中文)数据都转换为id表示的形式
        en_dict.get(w, 0) 默认值为0
        以英文句子长度排序的(句子下标)顺序为基准
        """
        out_en_ids = [[en_dict.get(w, 0) for w in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(w, 0) for w in sent] for sent in cn]

        # 构建一个按照句子长度排序的函数
        def len_argsort(seq):
            """
            传入一系列句子数据(分好词的列表形式)，
            按照句子长度排序后，返回排序后原来各句子在数据中的索引下标
            """
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        # 把中文和英文按照同样的顺序排序
        if sort:
            # 以英文句子长度排序的(句子下标)顺序为基准
            sorted_index = len_argsort(out_en_ids)

            # TODO: 对翻译前(英文)数据和翻译后(中文)数据都按此基准进行排序
            out_en_ids = [out_en_ids[i] for i in sorted_index]
            out_cn_ids = [out_cn_ids[i] for i in sorted_index]

        return out_en_ids, out_cn_ids

    def splitBatch(self, en, cn, batch_size, shuffle=True):
        """
        将以单词id列表表示的翻译前(英文)数据和翻译后(中文)数据
        按照指定的batch_size进行划分
        如果shuffle参数为True，则会对这些batch数据顺序进行随机打乱
        """
        # 在按数据长度生成的各条数据下标列表[0, 1, ..., len(en)-1]中
        # 每隔指定长度(batch_size)取一个下标作为后续生成batch的起始下标
        """
        BATCH_SIZE = 64  # 每批次训练数据数量
        idx_list :[    0    64   128   192   256    ...]
        利用np.random.shuffle将这些各batch起始下标打乱
        """
        idx_list = np.arange(0, len(en), batch_size)
        #
        if shuffle:
            np.random.shuffle(idx_list)
        # 存放各个batch批次的句子数据索引下标
        batch_indexs = []
        for idx in idx_list:

            """
            注意，起始下标最大的那个batch可能会超出数据大小
            因此要限定其终止下标不能超过数据大小
             min(idx + batch_size, len(en))
batch_indexs的数据如下
[[10368 10369 10370 10371 10372 10373 10374 10375 10376 10377 10378 10379
 10380 10381 10382 10383 10384 10385 10386 10387 10388 10389 10390 10391
 10392 10393 10394 10395 10396 10397 10398 10399 10400 10401 10402 10403
 10404 10405 10406 10407 10408 10409 10410 10411 10412 10413 10414 10415
 10416 10417 10418 10419 10420 10421 10422 10423 10424 10425 10426 10427
 10428 10429 10430 10431],[...]]
            """
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en))))


        batches = []
        for batch_index in batch_indexs:
            """
            按各batch批次的句子数据索引下标，构建实际的单词id列表表示的各batch句子数据
            batch_cn:
            [[  2 198  21 ...   0   0   0]
             [  2   5 296 ...   0   0   0]
             [  2  15  72 ...   0   0   0]
             ...
             [  2   5  13 ...   0   0   0]
             [  2 297 100 ...   0   0   0]
             [  2  11  32 ...  32   4   3]]
             使用Batch类将batch_en和batch_cn封装起来
            """
            # 按当前batch的各句子下标(数组批量索引)提取对应的单词id列表句子表示数据
            batch_en = [en[index] for index in batch_index]
            batch_cn = [cn[index] for index in batch_index]
            # 对当前batch的各个句子都进行padding对齐长度
            # 维度为：batch数量×batch_size×每个batch最大句子长度
            batch_cn = seq_padding(batch_cn)
            batch_en = seq_padding(batch_en)
            # 将当前batch的英文和中文数据添加到存放所有batch数据的列表中
            batches.append(Batch(batch_en, batch_cn))

        return batches


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        # trg_old = trg

        """
        将输入与输出的单词id表示的数据规范成整数类型
        src: [64, 6]
        trg: [64, 14]
        self.src_mask: [64(batch_size), 1, src.shape[1](列表中最长句子的长度)]
        != PAD的位置设成文=True，否则设成False
        tensor([[[True, True, True, True, True, True]],
                [[True, True, True, True, True, True]],

                [[True, True, True, True, True, True]],

                [[True, True, True, True, True, True]],

                [[True, True, True, True, True, True]],

                [[True, True, True, True, True, True]],

                [[True, True, True, True, True, True]],

                [[True, True, True, True, True, True]],

                [[True, True, True, True, True, True]],

                [[True, True, True, True, True, True]],

                [[True, True, True, True, True, True]],

                [[True, True, True, True, True, True]],

                [[True, True, True, True, True, True]],

                [[True, True, True, True, True, True]],

                [[True, True, True, True, True, True]],

                [[True, True, True, True, True, True]],

                [[True, True, True, True, True, True]],

                [[True, True, True, True, True, True]],

                [[True, True, True, True, True, True]],

                [[True, True, True, True, True, True]],

                [[True, True, True, True, True, True]]
        """
        src = torch.from_numpy(src).to(DEVICE).long()
        trg = torch.from_numpy(trg).to(DEVICE).long()

        self.src = src
        # 对于当前输入的句子非空部分进行判断成bool序列
        # 并在seq length前面增加一维，形成维度为 1×seq length 的矩阵
        self.src_mask = (src != pad).unsqueeze(-2)  # 64 x 1 x 19

        """
        如果输出目标不为空，则需要对decoder要使用到的target句子进行mask
        
        self.trg = trg[:, :-1] -> decoder要用到的target输入部分(不包含终止符号[EOS])
        self.trg_y = trg[:, 1:]  -> decoder训练时应预测输出的target结果(不包含开始符号[BOS])
        假如说trg为[64,19] 那 self.trg(去掉的是最后一列) 和 self.trg_y(去掉的是第一列) 都是 [64, 18]
        
        self.ntokens = (self.trg_y != pad).data.sum() 统计的是有效的单词数量
        假如说 self.trg_y.shape = [64,14] -> 64x14 = 896个单词 self.ntokens = 593 说明除593外的都是有效单词
        """

        if trg is not None:
            # decoder要用到的target输入部分
            self.trg = trg[:, :-1] #[batch_size, trg_len - 1]

            # decoder训练时应预测输出的target结果
            self.trg_y = trg[:, 1:] #[batch_size, trg_len - 1]

            # 将target输入部分进行attention mask
            self.trg_mask = self.make_std_mask(self.trg, pad) #64x18 x18
            # 将应输出的target结果中实际的词数进行统计

            self.ntokens = (self.trg_y != pad).data.sum()

    # Mask掩码操作
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."

        """
        tgt_mask = (tgt != pad).unsqueeze(-2) #[batch_size, 1, tgt_len]
        tgt_mask [[[ True,  True,  True,  ..., False, False, False]],

        [[ True,  True,  True,  ..., False, False, False]],

        [[ True,  True,  True,  ...,  True,  True,  True]],

        ...,

        [[ True,  True,  True,  ..., False, False, False]],

        [[ True,  True,  True,  ..., False, False, False]],

        [[ True,  True,  True,  ..., False, False, False]]]
        
subsequent_mask(tgt.size(-1)):(1, tgt_len, tgt_len)     
        [[0 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [0 0 1 1 1 1 1 1 1 1 1 1 1 1 1]
 [0 0 0 1 1 1 1 1 1 1 1 1 1 1 1]
 [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1]
 [0 0 0 0 0 1 1 1 1 1 1 1 1 1 1]
 [0 0 0 0 0 0 1 1 1 1 1 1 1 1 1]
 [0 0 0 0 0 0 0 1 1 1 1 1 1 1 1]
 [0 0 0 0 0 0 0 0 1 1 1 1 1 1 1]
 [0 0 0 0 0 0 0 0 0 1 1 1 1 1 1]
 [0 0 0 0 0 0 0 0 0 0 1 1 1 1 1]
 [0 0 0 0 0 0 0 0 0 0 0 1 1 1 1]
 [0 0 0 0 0 0 0 0 0 0 0 0 1 1 1]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 1 1]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
0对应的位置为True
tensor([[[ True, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False, False, False, False],
         [ True,  True, False, False, False, False, False, False, False, False,
          False, False, False, False, False, False, False, False],
         [ True,  True,  True, False, False, False, False, False, False, False,
          False, False, False, False, False, False, False, False],
         [ True,  True,  True,  True, False, False, False, False, False, False,
          False, False, False, False, False, False, False, False],
         [ True,  True,  True,  True,  True, False, False, False, False, False,
          False, False, False, False, False, False, False, False],
         [ True,  True,  True,  True,  True,  True, False, False, False, False,
          False, False, False, False, False, False, False, False],
         [ True,  True,  True,  True,  True,  True,  True, False, False, False,
          False, False, False, False, False, False, False, False],
         [ True,  True,  True,  True,  True,  True,  True,  True, False, False,
          False, False, False, False, False, False, False, False],
         [ True,  True,  True,  True,  True,  True,  True,  True,  True, False,
          False, False, False, False, False, False, False, False],
         [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          False, False, False, False, False, False, False, False],
         [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
           True, False, False, False, False, False, False, False],
         [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
           True,  True, False, False, False, False, False, False],
         [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
           True,  True,  True, False, False, False, False, False],
         [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
           True,  True,  True,  True, False, False, False, False],
         [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
           True,  True,  True,  True,  True, False, False, False],
         [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
           True,  True,  True,  True,  True,  True, False, False],
         [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
           True,  True,  True,  True,  True,  True,  True, False],
         [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
           True,  True,  True,  True,  True,  True,  True,  True]]])
        """

        tgt_mask = (tgt != pad).unsqueeze(-2) #[batch_size, 1, tgt_len]

        # test_0 = (tgt != pad)
        # test_1 = tgt.size(-1)
        # test_2 = subsequent_mask(test_1)
        # # test_3 = tgt_mask.numpy()
        test_4 = subsequent_mask(tgt.size(-1))
        # test_5 = tgt_mask.cpu().numpy()

        """
        [batch_size, 1, tgt_len-1] & [1, tgt_len-1, tgt_len-1] = [batch_size, tgt_len-1, tgt_len-1]
        每句话的每个位置都有一个Mask
        [64,17,17]
        """
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)) #[batch_size, tgt_len, tgt_len]

        # tgt_mask_cpu = tgt_mask.cpu().numpy()

        return tgt_mask


"""
获取每一个单词的词向量
"""
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # Embedding层
        self.lut = nn.Embedding(vocab, d_model)
        # Embedding维数
        self.d_model = d_model

    def forward(self, x):
        # 返回x对应的embedding矩阵（需要乘以math.sqrt(d_model)）
        return self.lut(x) * math.sqrt(self.d_model)


# 导入依赖库
import matplotlib.pyplot as plt
import seaborn as sns

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个size为 max_len(设定的最大长度)×embedding维度 的全零矩阵
        # 来存放所有小于这个长度位置对应的porisional embedding
        pe = torch.zeros(max_len, d_model, device=DEVICE)
        # 生成一个位置下标的tensor矩阵(每一行都是一个位置下标)
        """
        形式如：
        tensor([[0.],
                [1.],
                [2.],
                [3.],
                [4.],
                ...])
                
        PE(pos,2i)=sin(pos/10000^(2i/dmodel)) PE(pos,2i+1)=cos(pos/10000^(2i/dmodel))
        pe[:, 0::2] = torch.sin(position * div_term) 偶数0,2,4,6,8,10
        pe[:, 1::2] = torch.cos(position * div_term) 奇数1,3,5,7,9,11
        """
        position = torch.arange(0., max_len, device=DEVICE).unsqueeze(1) #[max_len, 1]
        # 这里幂运算太多，我们使用exp和log来转换实现公式中pos下面要除以的分母（由于是分母，要注意带负号）
        div_term = torch.exp(torch.arange(0., d_model, 2, device=DEVICE) * -(math.log(10000.0) / d_model)) #[d_model / 2]
        # div_term = div_term.unsqueeze(0)
        # div_term_numpy = div_term.cpu().numpy()

        # TODO: 根据公式，计算各个位置在各embedding维度上的位置纹理值，存放到pe矩阵中
        test_a = position * div_term
        test_a = test_a.cpu().numpy()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 加1个维度，使得pe维度变为：1×max_len×embedding维度
        # (方便后续与一个batch的句子所有词的embedding批量相加)

        """
        [1,5000,256]
        """
        pe = pe.unsqueeze(0)
        # 将pe矩阵以持久的buffer状态存下(不会作为要训练的参数)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将一个batch的句子所有词的embedding与已构建好的positional embeding相加
        # (这里按照该批次数据的最大句子长度来取对应需要的那些positional embedding值)
        # self.encoder(self.src_embed(src), src_mask)

        test_1 = x.size(1)
        test_2 = self.pe[:, :x.size(1)]
        test_2 = test_2.cpu().numpy()
        """
        x.size(1):9 可以看做是句子的长度
        self.pe[:, :x.size(1)]:  从5000个句子中取9行 -> [1,9,256] 
        
        [64 9 256] + [1,9, 256] //每个句子都加入PosEmbedding信息
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        来自self.encoder(self.src_embed(src), src_mask)	
        """

        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

# 可见，这里首先是按照最大长度max_len生成一个位置，而后根据公式计算出所有的向量，在forward函数中根据长度取用即可，非常方便。
#
#     注意要设置requires_grad=False，因其不参与训练。
#
# 下面画一下位置嵌入，可见纵向观察，随着embedding dimensionembedding dimension增大，位置嵌入函数呈现不同的周期变化。

# pe = PositionalEncoding(16, 0, 100)
# positional_encoding = pe.forward(Variable(torch.zeros(1, 100, 16, device=DEVICE)))
# plt.figure(figsize=(10,10))
# sns.heatmap(positional_encoding.squeeze().cpu().numpy())
# plt.title("Sinusoidal Function")
# plt.xlabel("hidden dimension")
# plt.ylabel("sequence length")
#
# plt.figure(figsize=(15, 5))
# pe = PositionalEncoding(20, 0)
# y = pe.forward(Variable(torch.zeros(1, 100, 20,device=DEVICE))).cpu()
# plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
# plt.legend(["dim %d"%p for p in [4,5,6,7]])


def attention(query, key, value, mask=None, dropout=None):
    # 将query矩阵的最后一个维度值作为d_k
    d_k = query.size(-1)

    """
    d_k:32
    query:        [64,8,9,32]
    key.transpose:[64,8,32,9]
    torch.matmul(query, key.transpose(-2, -1)):[64,8,9,9]
    
    mask:torch.Size([64, 1, 1, 9])
    如果存在要进行mask的内容，则将那些为0的部分替换成一个很大的负数
    
    torch.matmul(p_attn, value): [64,8,9,9]x[64,8,9,32] = [64,8, 9,32]
    """

    # TODO: 将key的最后两个维度互换(转置)，才能与query矩阵相乘，乘完了还要除以d_k开根号
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 如果存在要进行mask的内容，则将那些为0的部分替换成一个很大的负数
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # TODO: 将mask后的attention矩阵按照最后一个维度进行softmax
    p_attn = F.softmax(scores, dim = -1)

    # 如果dropout参数设置为非空，则进行dropout操作
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 最后返回注意力矩阵跟value的乘积，以及注意力矩阵
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()

        """
        d_model: 256
        h:       8
        self.d_k = d_model // h = #256 / 8 = 32 ->得 到一个head的attention表示维度
        定义4个全连接函数，供后续作为WQ，WK，WV矩阵和最后h个多头注意力矩阵concat之后进行变换的矩阵 为什么是4个?
        
        第四个矩阵是用来做线性变换self.linears[-1](x)
        """
        # 保证可以整除
        assert d_model % h == 0
        # 得到一个head的attention表示维度
        self.d_k = d_model // h #256 / 8 = 32
        # head数量
        self.h = h
        # 定义4个全连接函数，供后续作为WQ，WK，WV矩阵和最后h个多头注意力矩阵concat之后进行变换的矩阵
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        tag_mask:原本为[64,17,17]
        经过mask.unsqueeze(1) -> [64,1,17,17]
        """
        if mask is not None:
            mask = mask.unsqueeze(1)

        """
        mask:    torch.Size([64, 1, 1, 9])
        nbatches: batch_size
        query, key, value 计算前: [64,9, 256]
        query, key, value :[64(batch_size),8(head数量),9(单词的数量),32(每个head的维度)]
        交换位置是为了方便计算单词与单词之间的attention
        
        将h个多头注意力矩阵concat起来（注意要先把h变回到第三维的位置）
        """

        # query的第一个维度值为batch size
        nbatches = query.size(0)
        # 将embedding层乘以WQ，WK，WV矩阵(均为全连接)
        # 并将结果拆成h块，然后将第二个和第三个维度值互换(具体过程见上述解析)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        # 调用上述定义的attention函数计算得到h个注意力矩阵跟value的乘积，以及注意力矩阵
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 将h个多头注意力矩阵concat起来（注意要先把h变回到第三维的位置）
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 使用self.linears中构造的最后一个全连接函数来存放变换后的矩阵进行返回
        return self.linears[-1](x)

def clones(module, N):
    """
    克隆模型块，克隆的模型块参数不共享
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# FeedForwardFeedForward（前馈网络）层其实就是两层线性映射并用激活函数激活
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # TODO: 请利用init中的成员变量实现Feed Forward层的功能
        """
        self.w_1(x):  [64, 9, 1024] = [batch_size,src_len,d_model] * [d_model, d_ff] = [batch_size,src_len,d_ff]
        self.w_2(self.dropout(F.relu(self.w_1(x)))) -> [batch_size, src_len, d_models]
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Encoder(nn.Module):
    # layer = EncoderLayer
    # N = 6
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        """
        6个Encoder单元 
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        layer.size: dmodel
        """
        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # Layer Norm
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        使用循环连续eecode N次(这里为6次)
        这里的Eecoderlayer会接收一个对于输入的attention mask处理

        def encode(self, src, src_mask):
            return self.encoder(self.src_embed(src), src_mask)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()

        """
        self.size是dmodel
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        """
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # SublayerConnection的作用就是把multi和ffn连在一起
        # 只不过每一层输出之后都要先做Layer Norm再残差连接
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # d_model
        self.size = size

    def forward(self, x, mask):
        # 将embedding层进行Multi head Attention
        """
        return x + self.dropout(sublayer(self.norm(x)))
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 注意到attn得到的结果x直接作为了下一层的输入
        return self.sublayer[1](x, self.feed_forward)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 初始化α为全1, 而β为全0
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        # 平滑项
        self.eps = eps

    def forward(self, x):
        # TODO: 请利用init中的成员变量实现LayerNorm层的功能

        """
        y = \gamma \;(\frac{x-\mu(x)}{\sigma(x)}) + \beta\\
        """
        # 按最后一个维度计算均值和方差
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)

        # TODO: 返回Layer Norm的结果
        return self.a_2 * ( x - mean) / torch.sqrt(std ** 2 + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    SublayerConnection的作用就是把Multi-Head Attention和Feed Forward层连在一起
    只不过每一层输出之后都要先做Layer Norm再残差连接

    SublayerConnection(size, dropout)
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # TODO: 请利用init中的成员变量实现LayerNorm和残差连接的功能
        # 返回Layer Norm和残差连接后结果
        return x + self.dropout(sublayer(self.norm(x)))

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        # TODO: 参照EncoderLayer完成成员变量定义

        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # Layer Norm
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        使用循环连续decode N次(这里为6次)
        这里的Decoderlayer会接收一个对于输入的attention mask处理
        和一个对输出的attention mask + subsequent mask处理
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        # Self-Attention
        self.self_attn = self_attn
        # 与Encoder传入的Context进行Attention
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 用m来存放encoder的最终hidden表示结果
        m = memory

        """
        比Encoder多了一个Encoder-Decoder Attention
        DecodeAttention里面d的q,v矩阵是Encoder的输出值
        """
        # TODO: 参照EncoderLayer完成DecoderLayer的forwark函数
        # Self-Attention：注意self-attention的q，k和v均为decoder hidden
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Context-Attention：注意context-attention的q为decoder hidden，而k和v为encoder hidden
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    # 设定subsequent_mask矩阵的shape
    attn_shape = (1, size, size)

    # TODO: 生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵

    """
    subsequent_mask
    [[0 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
     [0 0 1 1 1 1 1 1 1 1 1 1 1 1 1]
     [0 0 0 1 1 1 1 1 1 1 1 1 1 1 1]
     [0 0 0 0 1 1 1 1 1 1 1 1 1 1 1]
     [0 0 0 0 0 1 1 1 1 1 1 1 1 1 1]
     [0 0 0 0 0 0 1 1 1 1 1 1 1 1 1]
     [0 0 0 0 0 0 0 1 1 1 1 1 1 1 1]
     [0 0 0 0 0 0 0 0 1 1 1 1 1 1 1]
     [0 0 0 0 0 0 0 0 0 1 1 1 1 1 1]
     [0 0 0 0 0 0 0 0 0 0 1 1 1 1 1]
     [0 0 0 0 0 0 0 0 0 0 0 1 1 1 1]
     [0 0 0 0 0 0 0 0 0 0 0 0 1 1 1]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 1 1]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
    """
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # test_1 = torch.from_numpy(subsequent_mask) == 0

    # TODO: 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return torch.from_numpy(subsequent_mask) == 0

# plt.figure(figsize=(5,5))
# plt.imshow(subsequent_mask(20)[0])
# plt.show()

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        self.encode
        self.decode
        """
        # encoder的结果作为decoder的memory参数传入，进行decode
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

class Generator(nn.Module):
    # vocab: tgt_vocab
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # decode后的结果，先进入一个全连接层变为词典大小的向量
        self.proj = nn.Linear(d_model, vocab)

    """
    decode后的结果，先进入一个全连接层变为词典大小的向量
    然后再进行log_softmax操作(在softmax结果上再做多一次log运算)
    """
    def forward(self, x):

        return F.log_softmax(self.proj(x), dim=-1)

"""
src_vocab： 原语料库的大小(单词的数量)
tgt_vocab： 目标语料库的大小
LAYERS = 6  # transformer中堆叠的encoder和decoder block层数
D_MODEL = 256  # embedding维数
D_FF = 1024  # feed forward第一个全连接层维数
H_NUM = 8  # multihead attention hidden个数
DROPOUT = 0.1  # dropout比例
"""

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    克隆对象
    """
    c = copy.deepcopy

    """
    PositionwiseFeedForward: forward 得到的是[batch_size, src_len, d_model]
    
    decode后的结果，先进入一个全连接层变为词典大小的向量
    然后再进行log_softmax操作(在softmax结果上再做多一次log运算)
    
    """

    # 实例化Attention对象
    attn = MultiHeadedAttention(h, d_model).to(DEVICE)
    # 实例化FeedForward对象
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(DEVICE)
    # 实例化PositionalEncoding对象
    position = PositionalEncoding(d_model, dropout).to(DEVICE)
    # 实例化Transformer模型对象
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        nn.Sequential(Embeddings(d_model, src_vocab).to(DEVICE), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab).to(DEVICE), c(position)),
        Generator(d_model, tgt_vocab)).to(DEVICE)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            # 这里初始化采用的是nn.init.xavier_uniform
            nn.init.xavier_uniform_(p)
    return model.to(DEVICE)

# 五. 模型训练
class LabelSmoothing(nn.Module):
    """标签平滑处理"""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()

        """
        LabelSmoothing(tgt_vocab, padding_idx = 0, smoothing= 0.0)
        padding_idx = 0
        confidence = 1
        smoothing = 0
        size: 3194
        """

        self.criterion = nn.KLDivLoss(reduction='sum') #https://www.jianshu.com/p/579a0f4cbf24
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size

        """
        x.shape:      [1024,3194(vocab_size)]
        target.shape: [1024]
        true_dist.fill_之后变成了
        
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0')
        
        
        target.data.unsqueeze(1): [1024,1]
        """

        test1 = x.cpu().data.numpy()

        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))

        test2 = target.data.unsqueeze(1)
        test3 = true_dist.cpu().numpy()

        """
        对每行进行填充,index为target.data.unsqueeze(1), 用self.confidence
        一行中第target.data[index]的数据为self.confidence
        true_dist[:, self.padding_idx] = 0 ->  第一列的值为0
        """
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        test4 = true_dist.cpu().numpy()

        true_dist[:, self.padding_idx] = 0

        test5 = true_dist.cpu().numpy()
        test6 = target.data.cpu().numpy()

        """
        https://blog.csdn.net/york1996/article/details/102955270
        返回target.data==0 但不为0的所有索引
        x,true_dist: [1152, 3194]
        """
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:

            test6 = mask.squeeze().cpu().numpy()

            test7 = true_dist.cpu().numpy()

            true_dist.index_fill_(0, mask.squeeze(), 0.0) #https://www.jianshu.com/p/e568213c8501 0按行

            test8 = true_dist.cpu().numpy()

        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

# # Label smoothing的例子
# crit = LabelSmoothing(5, 0, 0.4)  # 设定一个ϵ=0.4
# predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
#                              [0, 0.2, 0.7, 0.1, 0],
#                              [0, 0.2, 0.7, 0.1, 0]])
# v = crit(Variable(predict.log()),
#          Variable(torch.LongTensor([2, 1, 0])))
# # Show the target distributions expected by the system.
# print(crit.true_dist)
# plt.imshow(crit.true_dist)
# plt.show()
#
# crit = LabelSmoothing(5, 0, 0.1)
# def loss(x):
#     d = x + 3 * 1
#     predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
#     #print(predict)
#     return crit(Variable(predict.log()), Variable(torch.LongTensor([1]))).item()
# plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
# plt.show()


class SimpleLossCompute:
    """
    简单的计算损失和进行参数反向传播更新训练的函数
    """

    def __init__(self, generator, criterion, opt=None):
        """
        SimpleLossCompute(model.generator, criterion, optimizer)
        """
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):

        """
        out(x): [batch_size, trg_len - 1,d_model]
        self.trg_y = trg[:, 1:]  #[batch_size, trg_len - 1]

        x = self.generator(x)
        # decode后的结果，先进入一个全连接层变为词典大小的向量
        self.proj = nn.Linear(d_model, vocab)
        return F.log_softmax(self.proj(x), dim=-1)

        torch.Size([1024(64x16), 3194])-》每个Id对应的概率
        torch.Size([1024]) = 64x16 -》每个单词的Id
        """
        x = self.generator(x)

        """
        test_1 = x.contiguous().view(-1, -> x.size(-1)) [1024(64x16), 3194]
        # test_2 = y.contiguous().view(-1) -> [1024]
        """

        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm.float()


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):

        """
        NoamOpt(
        D_MODEL:  model_size,
        1:        factor,
        2000:     warmup,
        optimizer:torch.optim.Adam(model.parameters(), lr=0, betas=(0.9,0.98), eps=1e-9))
        """

        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        """
        论文里面提到了他们用的优化器，是以$\beta_1=0.9、\beta_2=0.98$ 和 $\epsilon = 10^{−9}$ 的 $Adam$ 为基础，
        而后使用一种warmup的学习率调整方式来进行调节。  
        具体公式如下：  
          
        $$ lrate = d^{−0.5}_{model}⋅min(step\_num^{−0.5},\; step\_num⋅warmup\_steps^{−1.5})$$  
        
        基本上就是用一个固定的 $warmup\_steps$ **先进行学习率的线性增长（热身）**，而后到达 $warmup\_steps$ 之后会随着 $step\_num$ 的增长，
        以 $step\_num$（步数）的反平方根成比例地**逐渐减小它**，
        他们用的 $warmup\_steps = 4000$ ，这个可以针对不同的问题自己尝试。s
        """
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    # return NoamOpt(model.src_embed[0].d_model, 2, 4000,
    #                torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-7))

# Three settings of the lrate hyperparameters.
# opts = [NoamOpt(512, 1, 4000, None),
#         NoamOpt(512, 1, 8000, None),
#         NoamOpt(256, 1, 4000, None)]

# opts = [NoamOpt(512, 1, 4000, None)]
# plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
# plt.legend(["512:4000", "512:8000", "256:4000"])
# # plt.show()


def run_epoch(data, model, loss_compute, epoch):

    """
    run_epoch(data.train_data, model, SimpleLossCompute(model.generator, criterion, optimizer), epoch)
    """
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.

    for i , batch in enumerate(data):
        """
        src_mask = (src != pad).unsqueeze(-2)  # 64 x 1 x 19
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)) #[batch_size, tgt_len, tgt_len]
        
        out: [batch_size, trg_len - 1,d_model]
        self.trg_y = trg[:, 1:]  #[batch_size, trg_len - 1]
   
          
        """

        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        # test_1 = batch.trg_y
        # test_2 = batch.ntokens

        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch %d Batch: %d Loss: %f Tokens per Sec: %fs" % (epoch, i - 1, loss / batch.ntokens, (tokens.float() / elapsed / 1000.)))
            start = time.time()
            tokens = 0

    return total_loss / total_tokens


def train(data, model, criterion, optimizer):
    """
    训练并保存模型
    """
    # 初始化模型在dev集上的最优Loss为一个较大值
    best_dev_loss = 1e5

    for epoch in range(EPOCHS):
        # 模型训练
        model.train()
        run_epoch(data.train_data, model, SimpleLossCompute(model.generator, criterion, optimizer), epoch)
        model.eval()

        # 在dev集上进行loss评估
        print('>>>>> Evaluate')
        dev_loss = run_epoch(data.dev_data, model, SimpleLossCompute(model.generator, criterion, None), epoch)
        print('<<<<< Evaluate loss: %f' % dev_loss)

        # TODO: 如果当前epoch的模型在dev集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
        if dev_loss < best_dev_loss:
            torch.save(model.state_dict(), SAVE_FILE)
            best_dev_loss = dev_loss
        print()

# test_aa = np.arange(9).reshape(3,3)
# test_aa = torch.from_numpy(test_aa).to(DEVICE).long()
# test_aa = test_aa.unsqueeze(-2)
# test_aa = test_aa.cpu().numpy()
# print(test_aa)
# print('\n')
# print(test_aa.shape)
# print((test_aa[:,:-1]))
# print((test_aa[:,1:]))
# print((test_aa>5).sum())
# print(3 // 2)


# 数据预处理
"""
获取到数据
src_vocab： 原语料库的大小(单词的数量)
tgt_vocab： 目标语料库的大小
LAYERS = 6  # transformer中堆叠的encoder和decoder block层数
D_MODEL = 256  # embedding维数s
D_FF = 1024  # feed forward第一个全连接层维数
H_NUM = 8  # multihead attention hidden个数
DROPOUT = 0.1  # dropout比例
"""
data = PrepareData(TRAIN_FILE, DEV_FILE)
src_vocab = len(data.en_word_dict)
tgt_vocab = len(data.cn_word_dict)
print("src_vocab %d" % src_vocab)
print("tgt_vocab %d" % tgt_vocab)

# 初始化模型
model = make_model(
                    src_vocab,
                    tgt_vocab,
                    LAYERS,
                    D_MODEL,
                    D_FF,
                    H_NUM,
                    DROPOUT
                )

# 训练
print(">>>>>>> start train")
train_start = time.time()
criterion = LabelSmoothing(tgt_vocab, padding_idx = 0, smoothing= 0.0)
optimizer = NoamOpt(D_MODEL, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9,0.98), eps=1e-9))

train(data, model, criterion, optimizer)
print(f"<<<<<<< finished train, cost {time.time()-train_start:.4f} seconds")

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    传入一个训练好的模型，对指定数据进行预测

    srcs:
    tensor([[  2, 684, 219,   4,   3]], device='cuda:0')

    src_mask:
    tensor([[[True, True, True, True, True]]], device='cuda:0')

    memory.shape:[1,5,256]
    tensor([[[-0.8779,  0.5413, -0.4980,  ...,  1.2255, -0.5578,  0.4383],
         [-0.8189,  0.3560, -0.3527,  ...,  1.1341, -0.5992,  0.3934],
         [-0.6386,  0.2663, -0.3175,  ...,  1.0623, -0.5792,  0.3122],
         [-0.9151,  0.3019, -0.5100,  ...,  1.1343, -0.5595,  0.3246],
         [-1.0434,  0.2334, -0.6855,  ...,  1.0713, -0.4478,  0.4140]]],
       device='cuda:0')

    ys
    tensor([[2]], device='cuda:0')

    test_1
    tensor([[[1]]], device='cuda:0')

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    """

    # 先用encoder进行encode
    memory = model.encode(src, src_mask)
    # 初始化预测内容为1×1的tensor，填入开始符('BOS')的id，并将type设置为输入数据类型(LongTensor)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)

    # 遍历输出的长度下标
    for i in range(max_len-1):

        test_1 = subsequent_mask(ys.size(1)).type_as(src.data)


        # decode得到隐层表示
        out = model.decode(memory,
                           src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))

        """
        test_2 = out[:, -1] -> [1,256]
        prob = [1,256]x[256,3124] = [1,3124]
        """
        test_2 = out[:, -1]

        # 将隐藏表示转为对词典各词的log_softmax概率分布表示
        prob = model.generator(out[:, -1])
        # 获取当前位置最大概率的预测词id
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        # 将当前位置预测的字符id与之前的预测内容拼接起来
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


def evaluate(data, model):
    """
    在data上用训练好的模型进行预测，打印模型翻译结果
    """
    # 梯度清零
    with torch.no_grad():
        # 在data的英文数据长度上遍历下标
        for i in range(len(data.dev_en)):
            # TODO: 打印待翻译的英文句子
            en_sent = " ".join([data.en_index_dict[w] for w in  data.dev_en[i]])
            print("\n" + en_sent)

            # TODO: 打印对应的中文句子答案
            cn_sent =" ".join([data.cn_index_dict[w] for w in  data.dev_cn[i]])
            print("".join(cn_sent))

            # 将当前以单词id表示的英文句子数据转为tensor，并放如DEVICE中
            src = torch.from_numpy(np.array(data.dev_en[i])).long().to(DEVICE)
            # 增加一维
            src = src.unsqueeze(0)
            # 设置attention mask
            src_mask = (src != 0).unsqueeze(-2)
            # 用训练好的模型进行decode预测
            out = greedy_decode(model, src, src_mask, max_len=MAX_LENGTH, start_symbol=data.cn_word_dict["BOS"])
            # 初始化一个用于存放模型翻译结果句子单词的列表
            translation = []
            # 遍历翻译输出字符的下标（注意：开始符"BOS"的索引0不遍历）
            for j in range(1, out.size(1)):
                # 获取当前下标的输出字符
                sym = data.cn_index_dict[out[0, j].item()]
                # 如果输出字符不为'EOS'终止符，则添加到当前句子的翻译结果列表
                if sym != 'EOS':
                    translation.append(sym)
                # 否则终止遍历
                else:
                    break
            # 打印模型翻译输出的中文句子结果
            print("translation: %s" % " ".join(translation))

# 预测
# 加载模型
model.load_state_dict(torch.load(SAVE_FILE))
# 开始预测
print(">>>>>>> start evaluate")
evaluate_start  = time.time()
evaluate(data, model)
print(f"<<<<<<< finished evaluate, cost {time.time()-evaluate_start:.4f} seconds")