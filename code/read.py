import codecs
import re
from itertools import izip
import operator
import numpy as np  


num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')

def is_number(token):
    return bool(num_regex.match(token))

def create_vocab(domain, use_doc, maxlen=0, vocab_size=0):
    file_list = ['../data_preprocessed/%s/train/sentence.txt'%domain,
                 '../data_preprocessed/%s/test/sentence.txt'%domain]

    if use_doc:
        file_list.append('../data_doc/yelp_large/text.txt')
        file_list.append('../data_doc/electronics_large/text.txt')

    print 'Creating vocab ...'

    total_words, unique_words = 0, 0
    word_freqs = {}

    for f in file_list:
        top = 0
        fin = codecs.open(f, 'r', 'utf-8')
        for line in fin:
            words = line.split()
            if maxlen > 0 and len(words) > maxlen:
                continue

            for w in words:
                if not is_number(w):
                    try:
                        word_freqs[w] += 1
                    except KeyError:
                        unique_words += 1
                        word_freqs[w] = 1
                    total_words += 1
    fin.close()

    print ('  %i total words, %i unique words' % (total_words, unique_words))
    sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)

    vocab = {'<pad>':0, '<unk>':1, '<num>':2}
    index = len(vocab)
    for word, _ in sorted_word_freqs:
        vocab[word] = index
        index += 1
        if vocab_size > 0 and index > vocab_size + 2:
            break
    if vocab_size > 0:
        print (' keep the top %i words' % vocab_size)

    return vocab


def read_data(vocab, maxlen, path=None, domain=None, phase=None):
    if path:
        f = codecs.open(path)
    else:
        f = codecs.open('../data_preprocessed/%s/%s/sentence.txt'%(domain, phase))

    data = []

    num_hit, unk_hit, total = 0., 0., 0.
    maxlen_x = 0

    for row in f:
        indices = []
        tokens = row.strip().split()

        if maxlen > 0 and len(tokens) > maxlen:
            continue

        if len(tokens) == 0:
            indices.append(vocab['<unk>'])
            unk_hit += 1
        for word in tokens:
            if is_number(word):
                indices.append(vocab['<num>'])
                num_hit += 1
            elif word in vocab:
                indices.append(vocab[word])
            else:
                indices.append(vocab['<unk>'])
                unk_hit += 1
            total += 1

        data.append(indices)
        if maxlen_x < len(tokens):
            maxlen_x = len(tokens)

    f.close()

    return data, maxlen_x


def read_label(domain, phase):
    f_t = codecs.open('../data_preprocessed/%s/%s/target.txt'%(domain, phase))
    f_o = codecs.open('../data_preprocessed/%s/%s/opinion.txt'%(domain, phase))
    f_p = codecs.open('../data_preprocessed/%s/%s/target_polarity.txt'%(domain, phase))
   
    target_label = []
    op_label = []
    pol_label = []
    for t, o, p in zip(f_t, f_o, f_p):
        target_label.append([int(s) for s in t.strip().split()])
        op_label.append([int(s) for s in o.strip().split()])
        pol_label.append([int(s) for s in p.strip().split()])
       

    f_t.close()
    f_o.close()
    f_p.close()

    return target_label, op_label, pol_label


def read_label_doc(path):
    f = codecs.open(path)
    doc_label = []
    for line in f:
        score = float(line.strip())
        if score > 3:
            doc_label.append(0)
        elif score < 3:
            doc_label.append(1)
        else:
            doc_label.append(2)
    f.close()
    return doc_label


def prepare_data(domain, vocab_size, use_doc, maxlen=0):
    assert domain in ['res', 'lt', 'res_15']

    # use_doc = 1

    vocab = create_vocab(domain, use_doc, maxlen, vocab_size)

    train_x, train_maxlen = read_data(vocab, maxlen, domain=domain, phase='train')
    train_label_target, train_label_opinion, train_label_polarity = read_label(domain, 'train')
    
    test_x, test_maxlen = read_data(vocab, maxlen, domain=domain, phase='test')
    test_label_target, test_label_opinion, test_label_polarity = read_label(domain, 'test')

    overall_maxlen_aspect = max(train_maxlen, test_maxlen)

    doc_res_x, doc_res_y, doc_lt_x, doc_lt_y, doc_res_maxlen, doc_lt_maxlen = None, None, None, None, None, None
    if use_doc:
        doc_res_x, doc_res_maxlen = read_data(vocab, maxlen, path='../data_doc/yelp_large/text.txt')
        doc_res_y = read_label_doc('../data_doc/yelp_large/label.txt')
        doc_lt_x, doc_lt_maxlen = read_data(vocab, maxlen, path='../data_doc/electronics_large/text.txt')
        doc_lt_y = read_label_doc('../data_doc/electronics_large/label.txt')

    return train_x, train_label_target, train_label_opinion, train_label_polarity,\
           test_x, test_label_target, test_label_opinion, test_label_polarity,\
           vocab, overall_maxlen_aspect,\
           doc_res_x, doc_res_y, doc_lt_x, doc_lt_y, doc_res_maxlen, doc_lt_maxlen



def get_statistics(label, polarity=None):
    num = 0
    if polarity:
        count = {'pos':0, 'neg':0, 'neu':0, 'conf':0}
        polarity_map = {1: 'pos', 2: 'neg', 3: 'neu', 4: 'conf'}

    for i in range(len(label)):
        if polarity:
            assert len(label[i]) == len(polarity[i])
        for j in range(len(label[i])):
            if label[i][j] == 1:
                num += 1
                if polarity:
                    count[polarity_map[polarity[i][j]]]+=1

    if polarity:
        return num, count
    else:
        return num
