# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019/12/17 10:41
# @Author  : HENRY
# @Email   : mogaoding@163.com
# @File    : doc-extractor
# @Project : dlcp_hub
# @Software: PyCharm

import tensorflow as tf
import numpy as np
import pickle
import os
import re, sys
import time
from sklearn import preprocessing
from data import batch_yield2, tag2label, get_loss_weight
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib import learn
from Train import evaluate_

## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.80  # need ~700MB GPU memory

label2tag = {value: key for key, value in tag2label.items()}


class Predict(object):

    def __init__(self, vocab, tag2label, model_path, pad_dict):
        self.vocab = vocab
        self.tag2label = tag2label
        self.model_path = model_path
        self.pad_dict = pad_dict
        flags = tf.flags
        # 评估参数
        # flags.DEFINE_integer('batch_size', 64, 'Batch Size (default: 64)')
        flags.DEFINE_string('checkpoint_dir', self.model_path[:-6], 'Checkpoint directory from training run')
        self.FLAGS = flags.FLAGS
        self.sess = tf.Session(config=config)
        print(self.FLAGS.checkpoint_dir)
        self.checkpoint_file = tf.train.latest_checkpoint(self.FLAGS.checkpoint_dir)
        print('checkpoint_file: {}'.format(self.checkpoint_file))
        graph = tf.Graph()
        with graph.as_default():
            self.sess = tf.Session(config=config)
            saver = tf.train.import_meta_graph('{}.meta'.format(self.checkpoint_file))
            saver.restore(self.sess, self.checkpoint_file)
            # 通过名字从图中获取占位符
            self.input_x = graph.get_operation_by_name('placeholder/input_x').outputs[0]
            self.max_sentence_num = graph.get_operation_by_name('placeholder/max_sentence_num').outputs[0]
            self.max_sentence_length = graph.get_operation_by_name('placeholder/max_sentence_length').outputs[0]
            self.strlenlist = graph.get_operation_by_name('placeholder_1/strlenlist').outputs[0]
            self.placelist = graph.get_operation_by_name('placeholder_1/placelist').outputs[0]
            self.doclenlist = graph.get_operation_by_name('placeholder_1/doclenlist').outputs[0]
            self.labels = graph.get_operation_by_name('placeholder_1/labels').outputs[0]
            self.sequence_lengths = graph.get_operation_by_name('placeholder_1/sequence_lengths').outputs[0]

            # 我们想要评估的tensor
            self.transition_params = graph.get_collection("transition_params")[0]
            self.logits = graph.get_collection("logits")[0]
            print("self.logits:" + str(self.logits))

            print("Session started")

    def predict(self, devpath):
        test_html, test_num_batches = batch_yield2(devpath, self.vocab, tag2label, self.pad_dict["test_max_sent_len"],
                                                   self.pad_dict["test_max_sent_num"])
        for step, (path, inputs, labels, sentslenlist, placelist, docslenlist) in enumerate(test_html):
            sys.stdout.write(
                ' processing: {} batch / {} batches--path:{}.'.format(step + 1, test_num_batches, path) + '\r')
            step_num = test_num_batches + step + 1
            feed_dict = self.get_feed_dict(inputs, sentslenlist, placelist, docslenlist)
            # print("feed_dict:" + str(feed_dict))
            logits, transition_params = self.sess.run([self.logits, self.transition_params], feed_dict=feed_dict)
            # self.transition_params_train = pickle.load(open(self.model_path + "/transition_params"))

            label_list = []
            # print("logits:" + str(logits))
            # print("transition_params:" + str(transition_params))
            for logit in logits:
                viterbi_seq, _ = viterbi_decode(logit, transition_params)
                # print("logit:" + str(logit))
                label_list.append(viterbi_seq)
                path_result = path.split("/")[-1]
                print("writepath:" + self.model_path + "/result/" + path_result)
                resultpath = self.model_path + "/result"
                if not os.path.exists(resultpath): os.makedirs(resultpath)
                with open(resultpath + "/" + path_result, "w") as fp:
                    with open(path) as fpr:
                        labels = []
                        for index, line in enumerate(fpr.readlines()):
                            # print(line)
                            label = line.strip().split("\t")[-1]
                            labels.append(tag2label[label])
                            print("label:{}".format(tag2label[label]))
                            print("index:{}, label:{}, viterbi_seq[index]:{}, line:{}".format(index, tag2label[label],
                                                                                              viterbi_seq[index], line))
                            fp.write(line.strip() + "\t" + label2tag[(viterbi_seq[index])] + "\n")
                        print("labels:{}, viterbi_seq:{} ".format([labels], viterbi_seq))
                        print(evaluate_([labels], [viterbi_seq]))

    def get_feed_dict(self, inputs, sentslenlist, placelist, docslenlist, mode="test"):
        sentslen_array = np.array(sentslenlist).reshape(-1, 1)
        place_array = np.array(placelist).reshape(-1, 1)
        docslen_array = np.array(docslenlist).reshape(-1, 1)

        scalerfile = self.model_path + "/scaler"
        if (mode == "train"):
            self.scaler_sentslen = preprocessing.StandardScaler().fit(sentslen_array)
            self.scaler_place = preprocessing.StandardScaler().fit(place_array)
            self.scaler_docslen = preprocessing.StandardScaler().fit(docslen_array)
            pickle.dump(self.scaler_sentslen, open(scalerfile + "_sentslen", 'wb'))
            pickle.dump(self.scaler_place, open(scalerfile + "_sentslen", 'wb'))
            pickle.dump(self.scaler_docslen, open(scalerfile + "_sentslen", 'wb'))
        elif (mode == "test"):
            # self.scaler_sentslen = pickle.load(open(scalerfile+"_sentslen", 'rb'))
            # self.scaler_place = pickle.load(open(scalerfile+"_place", 'rb'))
            # self.scaler_docslen = pickle.load(open(scalerfile+"_docslen", 'rb'))
            self.scaler_sentslen = preprocessing.StandardScaler().fit(sentslen_array)
            self.scaler_place = preprocessing.StandardScaler().fit(place_array)
            self.scaler_docslen = preprocessing.StandardScaler().fit(docslen_array)

        feed_dict = {self.input_x: np.array(inputs),
                     self.max_sentence_length: self.pad_dict["test_max_sent_len"],
                     self.max_sentence_num: self.pad_dict["test_max_sent_num"],
                     self.strlenlist: self.scaler_sentslen.transform(sentslen_array).reshape([1, -1, 1]),
                     self.placelist: self.scaler_place.transform(place_array).reshape([1, -1, 1]),
                     self.doclenlist: self.scaler_docslen.transform(docslen_array).reshape([1, -1, 1]),
                     self.sequence_lengths: [len(inputs)]}
        return feed_dict


if __name__ == '__main__':
    pass
