# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019/12/17 10:41
# @Author  : HENRY
# @Email   : mogaoding@163.com
# @File    : doc-extractor
# @Project : dlcp_hub
# @Software: PyCharm
import pickle, os, random
import numpy as np

## tags, BIO
tag2label = {"O": 0,
             "B-AUTHOR": 1, "B-INFO": 4,
             "B-CONTENT": 2, "I-CONTENT": 3,
             "B-TITLE": 5, "B-RESOURCE": 6,
             "B-TIME": 7

             }
tag2weight = {
    "O": 1.0, "B-AUTHOR": 2.0,
    "B-CONTENT": 2.0, "I-CONTENT": 2.0,
    "B-TITLE": 3.0, "B-RESOURCE": 3.0,
    "B-TIME": 3.0, "B-INFO": 3.0,
}


def batch_yield2(trainpath, vocab, tag2label, sentlen, sentnum):
    data = []
    list = os.listdir(trainpath)  # 列出文件夹下所有的目录与文件

    htmls = []
    htmlssentslenlist = []
    htmlsplacelist = []
    htmlsdocslenlist = []
    htmlslabellist = []
    paths = []
    # html = []
    # htmlsentslenlist = []
    # htmlplacelist = []
    # htmldocslenlist = []
    # htmllabellist = []
    for i in range(0, len(list)):
        docs = []
        sentslenlist = []
        placelist = []
        docslenlist = []
        labellist = []
        path = os.path.join(trainpath, list[i])
        if os.path.isfile(path):
            # sys.stdout.write(' path: {} .'.format(list[i]) + '\r')
            with open(path, encoding='utf-8') as fr:
                lines = fr.readlines()
            with open(path, encoding='utf-8') as fr:
                lenfile = len(fr.read())
            i = 0
            for r, line in enumerate(lines):
                if i == 0:
                    i += 1
                    continue
                if len(line.split("\t")) == 2:
                    doc, label = line.strip().split("\t")
                    sents = doc.split("[。；？！?!.]")
                    sent2ids = []
                    for sent in sents:
                        sent2id = sentence2id(sent, vocab)
                        if (len(sent2id) > sentlen):
                            sent2id = sent2id[:sentlen]
                        else:
                            sent2id.extend([0] * (sentlen - len(sent2id)))
                        sent2ids.append(sent2id)
                    if (len(sent2ids) > sentnum):
                        sent2ids = sent2ids[:sentnum]
                    else:
                        sent2ids.extend([[0] * sentlen] * (sentnum - len(sent2ids)))
                    docs.append(sent2ids)
                    sentslenlist.append(len(doc))
                    placelist.append(1.0 * r / len(lines))
                    docslenlist.append(lenfile)
                    if label == "B-CONTNET":
                        label = "B-CONTENT"
                        print(path)
                    if label == "I-AUTHOR":
                        label = "B-AUTHOR"
                        print("I-AUTHOR:" + path)
                    if label == "I-CONTNET":
                        label = "I-CONTENT"
                        print("I-CONTNET:" + path)
                    if label == "I-INFO":
                        label = "B-INFO"
                        print("I-INFO:" + path)
                    labellist.append(tag2label[label.upper()])

            htmlssentslenlist.append(sentslenlist)
            htmlsplacelist.append(placelist)
            htmlsdocslenlist.append(docslenlist)
            htmlslabellist.append(labellist)
            htmls.append(docs)
            paths.append(path)
            # if (i % batchesize == 0):
            #     htmlssentslenlist.append(htmlsentslenlist)
            #     htmlsplacelist.append(htmlplacelist)
            #     htmlsdocslenlist.append(htmldocslenlist)
            #     htmlslabellist.append(htmllabellist)
            #     htmls.append(html)
            #     htmlsentslenlist=[]
            #     htmlplacelist=[]
            #     htmldocslenlist=[]
            #     htmllabellist=[]
            #     html=[]

    numbatches = len(htmls)
    return zip(paths, htmls, htmlslabellist, htmlssentslenlist, htmlsplacelist, htmlsdocslenlist), numbatches


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line.strip() is 'O':
            continue
        if line != '\n':
            # print(line)
            [char, label] = line.strip().split('\t')
            sent_.append(char)
            tag_.append(label)
        else:
            if sent_:
                data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data


def vocab_build(vocab_path, corpus_path, min_count):
    """

    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in str(sent_):
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id) + 1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def sentence2id(sent, word2id):
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path, id2word=False):
    """

    :param vocab_path: the path of the file 'word2id.pkl'
    :param id2word: default False, if True, return a dictionary that key is the id of the word and value is the word
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))

    if id2word:
        id2word_ = {}
        for word, id in word2id.items():
            id2word_[id] = word
        return id2word_

    return word2id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pre_trained_embedding(embedding_path):
    """

    :param embedding_path: the path of the embedding matrix whose name here is 'embedding.txt'
    :return: None. Saves the embedding matrix to 'data_path/pretrain_embedding.npy'
    """
    embedding_matrix = []
    with open(embedding_path, 'r', encoding='utf-8') as embedding_file:
        lines = embedding_file.readlines()
    for line in lines:
        embedding_array = line.split(' ', 1)[1]
        embedding_array = [float(number) for number in embedding_array.split(' ')]
        embedding_matrix.append(embedding_array)
    embedding_matrix = np.float32(embedding_matrix)
    np.save(os.path.join('.', 'data_path', 'pretrain_embedding'), embedding_matrix)


def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def get_loss_weight(label_lists):
    loss_weight = [[0.1]] * len(label_lists)
    label2tag = {}
    for key in tag2label:
        label2tag[str(tag2label[key])] = key
    for index_batch, label_list in enumerate(label_lists):
        loss_weight[index_batch] = [1] * len(label_list)
        for index, label in enumerate(label_list):
            loss_weight[index_batch][index] = tag2weight[label2tag[str(label)]]
    return loss_weight


# def get_labels_bool(label_lists, max_seq_len, batch_size):
#     index_to_gather = [[] * max_seq_len] * batch_size
#     tags = [1, 2, 3, 4, 5, 6, 7, 8]
#     for index, label_list in enumerate(label_lists):
#         for tag in tags:
#             if tag in label_list and tag in [7,8]:
#

def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels


if __name__ == '__main__':
    pre_trained_embedding(os.path.join('.', 'embedding.txt'))
