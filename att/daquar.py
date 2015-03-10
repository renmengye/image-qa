import re
import numpy as np
import sys
import os
import pickle as pkl

def prepare_data(
                qas,
                imgfeat,
                worddict,
                maxlen=None,
                n_words=10000,
                zero_pad=False):
    seqs = []
    feat_list = []
    for q in qas:
        seqs.append([worddict[w] if \
        worddict.has_key(w) else \
        worddict['UNK'] \
        for w in q[0].split()])
        feat_list.append(imgfeat[q[1]])
    lengths = [len(s) for s in seqs]
    maxlen = np.max(lengths)
    n_samples = len(seqs)
    x = np.zeros((maxlen, n_samples), dtype='int64')
    x_mask = np.zeros((maxlen, n_samples), dtype='float32')
    y = np.zeros((n_samples), dtype='int64')
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx]+1, idx] = 1.
        y[idx] = qas[idx][2]
    f = np.zeros((y.shape[0], 196, 512), dtype='float32')
    for idx, ff in enumerate(feat_list):
        f[idx] = feat_list[idx]
    return x, x_mask, f, y

def escapeNumber(line):
    line = re.sub('^21$', 'twenty_one', line)
    line = re.sub('^22$', 'twenty_two', line)
    line = re.sub('^23$', 'twenty_three', line)
    line = re.sub('^24$', 'twenty_four', line)
    line = re.sub('^25$', 'twenty_five', line)
    line = re.sub('^26$', 'twenty_six', line)
    line = re.sub('^27$', 'twenty_seven', line)
    line = re.sub('^28$', 'twenty_eight', line)
    line = re.sub('^29$', 'twenty_nine', line)
    line = re.sub('^30$', 'thirty', line)
    line = re.sub('^11$', 'eleven', line)
    line = re.sub('^12$', 'twelve', line)
    line = re.sub('^13$', 'thirteen', line)
    line = re.sub('^14$', 'fourteen', line)
    line = re.sub('^15$', 'fifteen', line)
    line = re.sub('^16$', 'sixteen', line)
    line = re.sub('^17$', 'seventeen', line)
    line = re.sub('^18$', 'eighteen', line)
    line = re.sub('^19$', 'nineteen', line)
    line = re.sub('^20$', 'twenty', line)
    line = re.sub('^10$', 'ten', line)
    line = re.sub('^0$', 'zero', line)
    line = re.sub('^1$', 'one', line)
    line = re.sub('^2$', 'two', line)
    line = re.sub('^3$', 'three', line)
    line = re.sub('^4$', 'four', line)
    line = re.sub('^5$', 'five', line)
    line = re.sub('^6$', 'six', line)
    line = re.sub('^7$', 'seven', line)
    line = re.sub('^8$', 'eight', line)
    line = re.sub('^9$', 'nine', line)
    return line

def train_valid_split(imgids):
    split = {}
    for i in imgids:
        split[i] = 1
    count = 0
    for i in split.keys():
        if count < len(split) / 10:
            split[i] = 0
        else:
            break
        count += 1
    return split

def data_split(data, imgids, split):
    td = []
    vd = []
    for (d, i) in zip(data, imgids):
        if split[i] == 0:
            vd.append(d)
        else:
            td.append(d)
    return (td, vd)

def extract_qa(lines):
    questions = []
    answers = []
    imgIds = []
    lineMax = 0
    for i in range(0, len(lines) / 2):
        n = i * 2
        match = re.search('image(\d+)', lines[n])
        number = int((re.search('\d+', match.group())).group())
        line = lines[n]
        line = re.sub(' in the image(\d+)( \?\s)?', '' , line)
        line = re.sub(' in this image(\d+)( \?\s)?', '' , line)
        line = re.sub(' on the image(\d+)( \?\s)?', '' , line)
        line = re.sub(' of the image(\d+)( \?\s)?', '' , line)
        line = re.sub(' in image(\d+)( \?\s)?', '' , line)
        line = re.sub(' image(\d+)( \?\s)?', '' , line)
        questions.append(line)
        answer = escapeNumber(re.sub('\s$', '', lines[n + 1]))
        answers.append(answer)

        # Important! Here image id is 0-based.
        imgIds.append(number - 1)
        l = len(questions[i].split())
        if l > lineMax: lineMax = l
    return (questions, answers, imgIds)

def build_dict(lines, keystart):
    # From word to number.
    word_dict = {'UNK': keystart}
    # From number to word, numbers need to minus one to convert to list indices.
    word_array = ['UNK']
    # Key is 1-based, 0 is reserved for sentence end.
    key = keystart + 1
    for i in range(0, len(lines)):
        line = lines[i].replace(',', '')
        words = line.split(' ')
        for j in range(0, len(words)):
            if not word_dict.has_key(words[j]):
                word_dict[words[j]] = key
                word_array.append(words[j])
                key += 1
            else:
                k = word_dict[words[j]]
    return  word_dict, word_array

def combine(questions, answers, imgids):
    combo = []
    for (q, a, i) in zip(questions, answers, imgids):
        combo.append((q, i, a))
    return combo

def lookup(answers, ansdict):
    ansids = []
    for ans in answers:
        if ansdict.has_key(ans):
            ansids.append(ansdict[ans])
        else:
            ansids.append(ansdict['UNK'])
    return ansids

def load_data(load_train=True,
            load_val=True,
            load_test=True,
            qapath='../../../data/mpi-qa/',
            imgpath='../../../data/nyu-depth/'):
    with open(imgpath+'hidden5_4_conv.pkl', 'rb') as f:
        imgfeat = pkl.load(f)
        imgfeat = np.array(imgfeat.todense())
        imgfeat = imgfeat.reshape(imgfeat.shape[0],14*14,512)

    if load_train or load_val:
        with open(qapath+'qa.37.raw.train.txt', 'r') as f:
            lines = f.readlines()
        
        # Make the first 10% as validation
        (questions, answers, imgids) = extract_qa(lines)
        split = train_valid_split(imgids)
        t_questions, v_questions = data_split(questions, imgids, split)
        t_answers, v_answers = data_split(answers, imgids, split)
        t_imgids, v_imgids = data_split(imgids, imgids, split)

        # Build a dictionary only for training questions.
        worddict, idict = build_dict(t_questions, 1)
        ansdict, iansdict = build_dict(t_answers, 0)
        
        train = (combine(\
            t_questions, lookup(t_answers, ansdict), t_imgids),\
            imgfeat)\
            if load_train else None
        valid = (combine(\
            v_questions, lookup(v_answers, ansdict), v_imgids),\
            imgfeat)\
        if load_val else None
    else:
        train = None
        valid = None
    if load_test:
        with open(qapath+'qa.37.raw.test.txt', 'r') as f:
            lines = f.readlines()
        (r_questions, r_answers, r_imgids) = extract_qa(lines)
        test = (combine(
            r_questions, lookup(r_answers, ansdict), r_imgids),\
            imgfeat)
    else:
        test = None

    with open(qapath+'ansdict.pkl', 'w+') as f:
        pkl.dump(ansdict, f)
    with open(qapath+'qdict.pkl', 'w+') as f:
        pkl.dump(worddict, f)

    return train, valid, test, worddict
