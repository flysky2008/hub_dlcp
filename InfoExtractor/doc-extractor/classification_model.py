# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019/12/17 10:41
# @Author  : HENRY
# @Email   : mogaoding@163.com
# @File    : doc-extractor
# @Project : dlcp_hub
# @Software: PyCharm

import tensorflow as tf
import doc_extractor_model
from tensorflow.contrib import rnn
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

import copy


class Classification(object):
    """[Implementation of Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)""""
    def __init__(self, doc2vecmodel, num_tags, optimizer, lr_pl, batchsize=1, hidden_size=10):
        self.doc2vec = doc2vecmodel.doc_vec
        docembedingsize = self.doc2vec.shape[1]
        self.doc2vec = tf.reshape(self.doc2vec, [batchsize, -1, docembedingsize])
        self.hidden_size = hidden_size
        self.num_tags = num_tags
        self.optimizer = optimizer
        self.lr_pl = lr_pl
        with tf.name_scope('placeholder'):
            self.strlenlist = tf.placeholder(tf.float32, [batchsize, None, 1], name='strlenlist')
            self.placelist = tf.placeholder(tf.float32, [batchsize, None, 1], name='placelist')
            self.doclenlist = tf.placeholder(tf.float32, [batchsize, None, 1], name='doclenlist')
            self.labels = tf.placeholder(tf.int32, [batchsize, None], name='labels')
            self.sequence_lengths = tf.placeholder(tf.int32, [None], name='sequence_lengths')
            self.loss_weight = tf.placeholder(dtype=tf.float32, shape=[1, None], name="loss_weight")
        # self.labels = tf.reshape(self.labels,[1,-1])

        self.classification()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def classification(self):
        with tf.name_scope("classification"):
            ph_num_rows = tf.shape(self.strlenlist)[1]
            # self.doc2vec = tf.tile(self.doc2vec, tf.pack([ph_num_rows, 1]))
            feature = tf.concat([self.doc2vec, self.strlenlist, self.placelist, self.doclenlist], 2)
            # feature = self.doc2vec
            self.featureRnn = self.BidirectionalGRUEncoder(feature, "featureRnn")
            with tf.variable_scope("proj"):
                W = tf.get_variable(name="W",
                                    shape=[2 * self.hidden_size, self.num_tags],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    dtype=tf.float32)
                self.W = W  # 记录W，以便后面add_summary中画图用

                b = tf.get_variable(name="b",
                                    shape=[self.num_tags],
                                    initializer=tf.zeros_initializer(),
                                    dtype=tf.float32)

                s = tf.shape(self.featureRnn)
                output = tf.reshape(self.featureRnn, [-1, 2 * self.hidden_size])
                pred = tf.matmul(output, W) + b

                self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])
                tf.add_to_collection('logits', self.logits)

    def loss_op(self):
        with tf.name_scope("loss_op"):
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                        tag_indices=self.labels,
                                                                        sequence_lengths=self.sequence_lengths)
            log_likelihood = tf.multiply(log_likelihood, self.loss_weight)
            tf.add_to_collection('transition_params', self.transition_params)
            self.log_likelihood_shape = tf.shape(log_likelihood)
            self.loss = -tf.reduce_mean(log_likelihood)

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            # 添加GATE_GRAPH设定，可更换为GATE_NONE与GATE_OP,
            # 其中GATE_GRAPH代表计算完毕所有参数梯度之后再进行回传，
            # GATE_NONE表示计算完一个参数进行一次回传，
            # GATE_OP表示每一个计算节点中的参数计算完毕梯度后进行回传。
            GATE_GRAPH = optim.GATE_GRAPH
            grads_and_vars = optim.compute_gradients(self.loss, gate_gradients=GATE_GRAPH)
            # grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            grads_and_vars_clip = []
            self.grads_and_vars = grads_and_vars
            for g, v in grads_and_vars:
                g = self.clip_norm(g, 1, tf.norm(g))
                grads_and_vars_clip.append([g, v])
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init_op(self):
        with tf.variable_scope("init_op"):
            self.init_op = tf.global_variables_initializer()

    def BidirectionalGRUEncoder(self, inputs, name):
        """

        :param inputs:
        :param name:
        :return:
        """
        # 输入inputs的shape是[batch_size*maxsent, max_time, voc_size]
        with tf.variable_scope(name):
            GRU_cell_fw = rnn.GRUCell(self.hidden_size)
            GRU_cell_bw = rnn.GRUCell(self.hidden_size)
            # fw_outputs和bw_outputs的size都是[batch_size, max_time, hidden_size]
            ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                                 cell_bw=GRU_cell_bw,
                                                                                 inputs=inputs,
                                                                                 sequence_length=HAN_model.length(
                                                                                     inputs),
                                                                                 dtype=tf.float32)
            # outputs的size是[batch_size*maxsent, max_time, hidden_size*2]
            outputs = tf.concat((fw_outputs, bw_outputs), 2)
            return outputs

    def clip_norm(self, g, c, n):
        """
        
        :param g:
        :param c:
        :param n:
        :return:
        """
        """Clip a tensor by norm.
        Arguments:
          g: gradient tensor to clip.
          c: clipping threshold.
          n: norm of gradient tensor.
        Returns:
          Clipped gradient tensor.
        """
        if c > 0:
            condition = n >= c
            then_expression = lambda: math_ops.scalar_mul(c / n, g)
            else_expression = lambda: g

            if isinstance(g, ops.Tensor):
                g_shape = copy.copy(g.get_shape())
            elif isinstance(g, ops.IndexedSlices):
                g_shape = copy.copy(g.dense_shape)
            condition = tf.convert_to_tensor(condition, dtype=tf.bool)
            g = tf.cond(condition, then_expression, else_expression)
            if isinstance(g, ops.Tensor):
                g.set_shape(g_shape)
            elif isinstance(g, ops.IndexedSlices):
                g._dense_shape = g_shape

        return g
