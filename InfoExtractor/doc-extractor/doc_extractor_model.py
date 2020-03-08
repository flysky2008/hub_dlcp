# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019/12/17 10:41
# @Author  : HENRY
# @Email   : mogaoding@163.com
# @File    : doc-extractor
# @Project : dlcp_hub
# @Software: PyCharm
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers


def length(sequences):
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    return tf.cast(seq_len, tf.int32)


class UniDocInfoExtractor(object):
    def __init__(self, vocab_size, embedding_size=10, hidden_size=10):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        with tf.name_scope('placeholder'):
            self.max_sentence_num = tf.placeholder(tf.int32, name='max_sentence_num')
            self.max_sentence_length = tf.placeholder(tf.int32, name='max_sentence_length')
            # self.batch_size = tf.placeholder(tf.int32, name='batch_size')
            # x的shape为[batch_size, 句子数， 句子长度(单词个数)]，但是每个样本的数据都不一样，，所以这里指定为空
            # y的shape为[batch_size, num_classes]
            self.input_x = tf.placeholder(tf.int32, [None, None, None], name='input_x')

        # 构建模型
        word_embedded, self.embedding_mat = self.word2vec()
        sent_vec = self.sent2vec(word_embedded)
        self.doc_vec = self.doc2vec(sent_vec)

    def word2vec(self):
        with tf.name_scope("embedding"):
            # tf.truncated_normal(shape, mean, stddev): shape表示生成张量的维度，mean是均值，stddev是标准差。这个函数产生正太分布，均值和标准差自己设定。
            embedding_mat = tf.Variable(tf.truncated_normal((self.vocab_size, self.embedding_size)), trainable=True)
            # shape为[batch_size, sent_in_doc, word_in_sent, embedding_size]
            word_embedded = tf.nn.embedding_lookup(embedding_mat, self.input_x)
        return word_embedded, embedding_mat

    def sent2vec(self, word_embedded):
        with tf.name_scope("sent2vec"):
            # GRU的输入tensor是[batch_size, max_time, ...].在构造句子向量时max_time应该是每个句子的长度，所以这里将
            # batch_size * sent_in_doc当做是batch_size.这样一来，每个GRU的cell处理的都是一个单词的词向量
            # 并最终将一句话中的所有单词的词向量融合（Attention）在一起形成句子向量

            # shape为[batch_size*sent_in_doc, word_in_sent, embedding_size]
            word_embedded = tf.reshape(word_embedded, [-1, self.max_sentence_length, self.embedding_size])
            # shape为[batch_size*sent_in_doce, word_in_sent, hidden_size*2]
            word_encoded = self.BidirectionalGRUEncoder(word_embedded, name='word_encoder')
            # shape为[batch_size*sent_in_doc, hidden_size*2]
            sent_vec = self.AttentionLayer(word_encoded, name='word_attention')
            return sent_vec

    def doc2vec(self, sent_vec):
        with tf.name_scope("doc2vec"):
            sent_vec = tf.reshape(sent_vec, [-1, self.max_sentence_num, self.hidden_size * 2])
            # shape为[batch_size, sent_in_doc, hidden_size*2]
            doc_encoded = self.BidirectionalGRUEncoder(sent_vec, name='sent_encoder')
            # shape为[batch_szie, hidden_szie*2]
            doc_vec = self.AttentionLayer(doc_encoded, name='sent_attention')
            return doc_vec

    def BidirectionalGRUEncoder(self, inputs, name):
        # 输入inputs的shape是[batch_size*maxsent, max_time, voc_size]
        with tf.variable_scope(name):
            GRU_cell_fw = rnn.GRUCell(self.hidden_size)
            GRU_cell_bw = rnn.GRUCell(self.hidden_size)
            # fw_outputs和bw_outputs的size都是[batch_size, max_time, hidden_size]
            ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                                 cell_bw=GRU_cell_bw,
                                                                                 inputs=inputs,
                                                                                 sequence_length=length(inputs),
                                                                                 dtype=tf.float32)
            # outputs的size是[batch_size*maxsent, max_time, hidden_size*2]
            outputs = tf.concat((fw_outputs, bw_outputs), 2)
            return outputs

    def AttentionLayer(self, inputs, name):
        # inputs是GRU的输出，size是[batch_size, max_time, encoder_size(hidden_size * 2)]
        with tf.variable_scope(name):
            # u_context是上下文的重要性向量，用于区分不同单词/句子对于句子/文档的重要程度,
            # 因为使用双向GRU，所以其长度为2×hidden_szie
            u_context = tf.Variable(tf.truncated_normal([self.hidden_size * 2]), name='u_context')
            # 使用一个全连接层编码GRU的输出的到期隐层表示,输出u的size是[batch_size*maxsent, max_time, hidden_size * 2]
            h = layers.fully_connected(inputs, self.hidden_size * 2, activation_fn=tf.nn.tanh)
            # shape为[batch_size, max_time, 1]
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keepdims=True), axis=1)
            # reduce_sum之前shape为[batch_szie*maxsent, max_time, hidden_szie*2]，之后shape为[batch_size*maxsent, hidden_size*2]
            atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
            return atten_output
