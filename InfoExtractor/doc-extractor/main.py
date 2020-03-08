# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019/12/17 10:41
# @Author  : HENRY
# @Email   : mogaoding@163.com
# @File    : doc-extractor
# @Project : dlcp_hub
# @Software: PyCharm
import os
import configparser
from data import read_dictionary, tag2label
from data_helper import build_vocab_doc
import time
from doc_extrator_model import UniDocInfoExtractor
from classification_model import classification
from Train import Operate
from Predict import Predict

iniFileUrl = os.path.join(os.path.abspath('.'), "main.ini")
conf = configparser.ConfigParser()  # 生成conf对象
conf.read(iniFileUrl, encoding='utf-8')  # 读取ini配置文件
# path配置
train_data_path = os.path.join(".", conf.get('path_arg', 'train_data_path'))
dev_data_path = os.path.join(".", conf.get('path_arg', 'dev_data_path'))
output_path = conf.get('path_arg', 'output_path')
# 训练参数配置
optimizer = conf.get('train_arg', 'optimizer')
lr_pl = float(conf.get('train_arg', 'lr_pl'))
# graph参数配置
hanModel_wordEmbedSzie = int(conf.get('graph_arg', 'uniDocModel_wordEmbedSize'))
hanModel_hiddenSize = int(conf.get('graph_arg', 'uniDocModel_hiddenSize'))
classModel_hiddenSzie = int(conf.get('graph_arg', 'classModel_hiddenSize'))
# pad参数配置
train_max_sent_len = int(conf.get('pad_arg', 'train_max_sent_len'))
train_max_sent_num = int(conf.get('pad_arg', 'train_max_sent_num'))
test_max_sent_len = int(conf.get('pad_arg', 'test_max_sent_len'))
test_max_sent_num = int(conf.get('pad_arg', 'test_max_sent_num'))

if __name__ == "__main__":
    mode = "train"
    if os.path.exists(os.path.join("data_path", 'word2id.pkl')):
        word2id = read_dictionary(os.path.join("data_path", 'word2id.pkl'))
    else:
        build_vocab_doc(os.path.join("data_path", 'word2id.pkl'), train_data_path)
        word2id = read_dictionary(os.path.join("data_path", 'word2id.pkl'))
    vocab_size = len(word2id)
    num_tags = len(tag2label)
    if mode == "train":
        timestamp = str(int(time.time()))
    else:
        timestamp = conf.get('path_arg', 'test_time')
    output_path = os.path.join(output_path, timestamp)
    if not os.path.exists(output_path): os.makedirs(output_path)
    model_path = os.path.join(output_path, conf.get('path_arg', 'model_path'))
    summary_path = os.path.join(output_path, conf.get('path_arg', 'summary_path'))
    log_path = os.path.join(output_path, conf.get('path_arg', 'log_path'))
    path_dict = {
        "model_path": model_path,
        "summary_path": summary_path,
        "log_path": log_path}
    pad_dict = {
        "train_max_sent_len": train_max_sent_len,
        "train_max_sent_num": train_max_sent_num,
        "test_max_sent_len": test_max_sent_len,
        "test_max_sent_num": test_max_sent_num
    }
    if mode == "train":
        doc2vecmodel = HAN(vocab_size, hanModel_wordEmbedSzie, hanModel_hiddenSize)
        classificalmodel = classification(doc2vecmodel, num_tags, optimizer, lr_pl)
        Operate = Operate(word2id, tag2label, doc2vecmodel, classificalmodel, path_dict, pad_dict)
        Operate.train(train_data_path, dev_data_path)
    if mode == "predict":
        predict = Predict(word2id, tag2label, model_path, pad_dict)
        predict.predict(dev_data_path)
