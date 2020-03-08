# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019/12/17 10:41
# @Author  : HENRY
# @Email   : mogaoding@163.com
# @File    : doc-extractor
# @Project : dlcp_hub
# @Software: PyCharm

import tensorflow as tf
from data import batch_yield2, get_loss_weight
import os, time, sys, re
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
from tensorflow.contrib.crf import viterbi_decode
from sklearn import preprocessing
from utils import get_logger
import pickle

projector_config = projector.ProjectorConfig()
embedding = projector_config.embeddings.add()
embedding.tensor_name = 'word embedding'
embedding.metadata_path = os.path.join(os.path.abspath('.'), 'metadata.tsv')
Title_pattern = re.compile('TITLE: precision:  (.*?)%; recall:  (.*?)%')
Resource_pattern = re.compile('RESOURCE: precision:  (.*?)%; recall:  (.*?)%')
Time_pattern = re.compile('TIME: precision:  (.*?)%; recall:  (.*?)%')
Author_pattern = re.compile('AUTHOR: precision:  (.*?)%; recall:  (.*?)%')
Content_pattern = re.compile('CONTENT: precision:  (.*?)%; recall:  (.*?)%')
Info_pattern = re.compile('INFO: precision:  (.*?)%; recall:  (.*?)%')

## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.80  # need ~700MB GPU memory


class Operate(object):
    def __init__(self, vocab, tag2label, doc2vecmodel, classificalmodel, path_dict, pad_dict):
        self.vocab = vocab
        self.tag2label = tag2label
        self.doc2vecmodel = doc2vecmodel
        self.classificalmodel = classificalmodel
        self.max_sentence_length = 60
        self.max_sentence_num = 30
        self.model_path = path_dict["model_path"]
        self.summary_path = path_dict["summary_path"]
        self.logger = get_logger(path_dict["log_path"])
        self.pad_dict = pad_dict
        self.correct = 0.0

    def train(self, trainpath, devpath):
        """

        :param train: 训练数据
        :param dev: 验证数据
        :return:
        """
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session(config=config) as sess:
            sess.run(self.classificalmodel.init_op)
            self.add_summary(sess)

            for epoch in range(10):
                self.run_one_epoch(sess, trainpath, devpath, self.tag2label, epoch, saver)

    def run_one_epoch(self, sess, trainpath, devpath, tag2label, epoch, saver):
        """

        :param sess: 与tensorflow后台对话的Session
        :param train: 训练数据
        :param dev: 验证数据
        :param tag2label: tag与label的对应关系
        :param epoch: 训练轮数
        :param saver: 存储训练结果的存储器
        :return:
        """
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        html, num_batches = batch_yield2(trainpath, self.vocab, tag2label, self.pad_dict["train_max_sent_len"],
                                         self.pad_dict["train_max_sent_num"])
        test_html, test_num_batches = batch_yield2(devpath, self.vocab, tag2label, self.pad_dict["test_max_sent_len"],
                                                   self.pad_dict["test_max_sent_num"])
        self.vars = tf.global_variables()

        for step, (path, inputs, labels, sentslenlist, placelist, docslenlist) in enumerate(html):
            sys.stdout.write(' processing: {} batch / {} batches--path:{}.'.format(step + 1, num_batches, path) + '\r')
            step_num = epoch * num_batches + step + 1
            feed_dict = self.get_feed_dict(inputs, sentslenlist, placelist, docslenlist, labellist=labels)
            _, self.transition_params, loss_train, summary, step_num_ = sess.run(
                [self.classificalmodel.train_op, self.classificalmodel.transition_params, self.classificalmodel.loss,
                 self.merged, self.classificalmodel.global_step],
                feed_dict=feed_dict)
            # pickle.dump(self.transition_params, open(self.model_path + "/transition_params", 'wb'))
            if step + 1 == 1 or (step + 1) % 1 == 0 or step + 1 == num_batches:
                self.logger.info(
                    '{} epoch {}, step {}, loss: {:.4}, global_step: {}, path: {}'.format(start_time, epoch + 1,
                                                                                          step + 1,
                                                                                          loss_train, step_num, path))
            self.file_writer.add_summary(summary, step_num_)
            # correct = self.evaluate(sess, test_html, test_num_batches, self.pad_dict["test_max_sent_len"],
            #                         self.pad_dict["test_max_sent_num"])
            # self.logger.info('correct: {}'.format(correct))
        correct, recall, precision = self.evaluate(sess, test_html, test_num_batches,
                                                   self.pad_dict["test_max_sent_len"],
                                                   self.pad_dict["test_max_sent_num"])
        self.logger.info('correct: {}'.format(correct))
        self.logger.info('recall: {}'.format(recall))
        self.logger.info('precision: {}'.format(precision))
        if self.correct < correct:
            self.correct = correct
            saver.save(sess, self.model_path, global_step=step_num)
            projector.visualize_embeddings(tf.summary.FileWriter(self.model_path), projector_config)

    def get_feed_dict(self, inputs, sentslenlist, placelist, docslenlist, labellist=None, mode="train"):
        sentslen_array = np.array(sentslenlist).reshape(-1, 1)
        place_array = np.array(placelist).reshape(-1, 1)
        docslen_array = np.array(docslenlist).reshape(-1, 1)
        if not os.path.exists(self.model_path): os.makedirs(self.model_path)
        scalerfile = self.model_path + "/scaler"
        if mode == "train":
            self.scaler_sentslen = preprocessing.StandardScaler().fit(sentslen_array)
            self.scaler_place = preprocessing.StandardScaler().fit(place_array)
            self.scaler_docslen = preprocessing.StandardScaler().fit(docslen_array)
            pickle.dump(self.scaler_sentslen, open(scalerfile + "_sentslen", 'wb'))
            pickle.dump(self.scaler_place, open(scalerfile + "_place", 'wb'))
            pickle.dump(self.scaler_docslen, open(scalerfile + "_docslen", 'wb'))
        elif mode == "test":
            self.scaler_sentslen = pickle.load(open(scalerfile + "_sentslen", 'rb'))
            self.scaler_place = pickle.load(open(scalerfile + "_place", 'rb'))
            self.scaler_docslen = pickle.load(open(scalerfile + "_docslen", 'rb'))

        feed_dict = {self.doc2vecmodel.input_x: np.array(inputs),
                     self.doc2vecmodel.max_sentence_length: self.max_sentence_length,
                     self.doc2vecmodel.max_sentence_num: self.max_sentence_num,
                     self.classificalmodel.strlenlist: self.scaler_sentslen.transform(sentslen_array).reshape(
                         [1, -1, 1]),
                     self.classificalmodel.placelist: self.scaler_place.transform(place_array).reshape([1, -1, 1]),
                     self.classificalmodel.doclenlist: self.scaler_docslen.transform(docslen_array).reshape([1, -1, 1]),
                     self.classificalmodel.sequence_lengths: [len(inputs)]}
        if labellist is not None:
            feed_dict[self.classificalmodel.labels] = np.array([labellist])
            feed_dict[self.classificalmodel.loss_weight] = get_loss_weight([labellist])
            # print(self.classificalmodel.loss_weight)
            # print("loss_weight:"+str(get_loss_weight([labellist])))
        return feed_dict

    def evaluate(self, sess, html, num_batches, max_sentence_num, max_sentence_length):
        all_score = 0.0
        all_recall = 0.0
        all_precision = 0.0
        for step, (path, inputs, labels, sentslenlist, placelist, docslenlist) in enumerate(html):
            predict_labels = self.predict_labels(sess, inputs, max_sentence_num, max_sentence_length, sentslenlist,
                                                 placelist, docslenlist)
            # print("labels_lem:"+str(len(labels)))
            score, recall, precision = self.evaluate_([labels], predict_labels)
            all_score += score
            all_recall += recall
            all_precision += precision
        return all_score / num_batches, all_recall / num_batches, all_precision / num_batches

    def predict_labels(self, sess, inputs, max_sentence_num, max_sentence_length, sentslenlist, placelist, docslenlist):
        feed_dict = self.get_feed_dict(inputs, sentslenlist, placelist, docslenlist)
        feed_dict[self.doc2vecmodel.max_sentence_num] = max_sentence_num
        feed_dict[self.doc2vecmodel.max_sentence_length] = max_sentence_length
        logits = sess.run(self.classificalmodel.logits, feed_dict=feed_dict)
        label_list = []
        # print("logits:" + str(logits))
        for logit in logits:
            viterbi_seq, _ = viterbi_decode(logit, self.transition_params)
            # print("logit:" + str(logit))
            label_list.append(viterbi_seq)
        return label_list

    def evaluate_(self, labels, predict_labels):
        print("labelss:" + str(predict_labels))
        labelss = []
        predict_labelss = []
        if len(labels) == len(predict_labels):
            for labels_evbatch, predict_labels_evbatch in zip(labels, predict_labels):
                labelss.extend(labels_evbatch)
                predict_labelss.extend(predict_labels_evbatch)
            # print("labelss:"+str(len(labelss)))
            # print("predict_labelss:" + str(len(predict_labelss)))
            if len(labelss) == len(predict_labelss):
                score = sum([1 for i, label in enumerate(labelss) if label == predict_labelss[i]]) * 1.0 / len(labelss)
                tp = sum([1 for i, label in enumerate(labelss) if (label != 0 and label == predict_labelss[i])]) * 1.0
                tpfn = sum([1 for label in labelss if (label != 0)]) * 1.0
                tpfp = sum([1 for label in predict_labelss if (label != 0)]) * 1.0
                if tpfn != 0:
                    recall = tp / tpfn
                else:
                    recall = 1
                if tpfp != 0:
                    precision = tp / tpfp
                else:
                    precision = 1
                return score, recall, precision
            else:
                print("生成的标签长度和原始标签长度不一致")
        else:
            print("生成的batchsize不一样")

    def add_summary(self, sess):
        """

        :param sess: 与tensorflow后台对话的Session
        :return:
        """
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)
        self.BiLSTM_forward_hidden_hist = tf.summary.histogram('BiLSTM_feature_bfward_hidden',
                                                               self.classificalmodel.featureRnn)
        self.loss_hist = tf.summary.histogram('loss_histogram', self.classificalmodel.loss)
        self.merged = tf.summary.merge_all()


def evaluate_(labels, predict_labels):
    # print("labelss:" + str(predict_labels))
    labelss = []
    predict_labelss = []
    if (len(labels) == len(predict_labels)):
        for labels_evbatch, predict_labels_evbatch in zip(labels, predict_labels):
            labelss.extend(labels_evbatch)
            predict_labelss.extend(predict_labels_evbatch)
        # print("labelss:"+str(len(labelss)))
        # print("predict_labelss:" + str(len(predict_labelss)))
        if (len(labelss) == len(predict_labelss)):
            score = sum([1 for i, label in enumerate(labelss) if label == predict_labelss[i]]) * 1.0 / len(labelss)
            tp = sum([1 for i, label in enumerate(labelss) if (label != 0 and label == predict_labelss[i])]) * 1.0
            tpfn = sum([1 for label in labelss if (label != 0)]) * 1.0
            tpfp = sum([1 for label in predict_labelss if (label != 0)]) * 1.0
            if (tpfn != 0):
                recall = tp / tpfn
            else:
                recall = 1
            if (tpfp != 0):
                precision = tp / tpfp
            else:
                precision = 1
            return score, recall, precision
        else:
            print("生成的标签长度和原始标签长度不一致")
    else:
        print("生成的batchsize不一样")


if __name__ == "__main__":
    pass
