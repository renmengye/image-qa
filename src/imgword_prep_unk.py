import numpy as np
import re
import sys
import os

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

def trainValidSplit(imgids):
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

def dataSplit(data, imgids, split):
    td = []
    vd = []
    for (d, i) in zip(data, imgids):
        if split[i] == 0:
            vd.append(d)
        else:
            td.append(d)
    return (td, vd)

def extractQA(lines):
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
        imgIds.append(number)
        l = len(questions[i].split())
        if l > lineMax: lineMax = l
    return (questions, answers, imgIds)

def buildDict(lines, keystart):
    # From word to number.
    word_dict = {}
    # From number to word, numbers need to minus one to convert to list indices.
    word_array = []
    # Word frequency
    word_freq = []
    # if key is 1-based, then 0 is reserved for sentence end.
    key = keystart

    for i in range(0, len(lines)):
        line = lines[i].replace(',', '')
        words = line.split(' ')
        for j in range(0, len(words)):
            if not word_dict.has_key(words[j]):
                word_dict[words[j]] = key
                word_array.append(words[j])
                word_freq.append(1)
                key += 1
            else:
                k = word_dict[words[j]]
                word_freq[k - 1] += 1
    word_dict['UNK'] = key
    word_array.append('UNK')
    return  word_dict, word_array

def lookupAnsID(answers, ansdict):
    ansids = []
    for ans in answers:
        if ansdict.has_key(ans):
            ansids.append(ansdict[ans])
        else:
            ansids.append(ansdict['UNK'])
    return np.array(ansids, dtype=int).reshape(len(ansids), 1)

def lookupQID(questions, worddict):
    wordslist = []
    maxlen = 27
    for q in questions:
        words = q.split(' ')
        wordslist.append(words)
        # if len(words) > maxlen:
        #     maxlen = len(words)
    result = np.zeros((len(questions), maxlen, 1), dtype=int)
    for i,words in enumerate(wordslist):
        for j,w in enumerate(words):
            if worddict.has_key(w):
                result[i, j, 0] = worddict[w]
            else:
                result[i, j, 0] = worddict['UNK']
    return result

def combine(wordids, imgids):
    return np.concatenate(\
        (np.array(imgids).reshape(len(imgids), 1, 1), \
        wordids), axis=1)

def combineAttention(wordids, imgids):
    imgid_t = []
    for n in range(0, wordids.shape[0]):
        for t in range(0, wordids.shape[1]):
            if wordids[n,t] == 0:
                imgid_t.append(0)
            else:
                imgid_t.append(imgids[n])

    return np.concatenate(
            (np.array(imgid_t).reshape(len(imgids), wordids.shape[1], 1),
            wordids), axis=-1)

if __name__ == '__main__':
    """
    Usage: imgword_prep.py -train trainQAFile -test testQAFile -o outputFolder
    """
    if len(sys.argv) > 6:
        for i in range(1, len(sys.argv)):
            if sys.argv[i] == '-train':
                trainQAFilename = sys.argv[i + 1]
            elif sys.argv[i] == '-test':
                testQAFilename = sys.argv[i + 1]
            elif sys.argv[i] == '-o':
                outputFolder = sys.argv[i + 1]
    else:
        trainQAFilename = '../../../data/mpi-qa/qa.37.raw.train.txt'
        testQAFilename = '../../../data/mpi-qa/qa.37.raw.test.txt'
        outputFolder = '../data/imgword'

    with open(trainQAFilename) as f:
        lines = f.readlines()

    (questions, answers, imgids) = extractQA(lines)
    split = trainValidSplit(imgids)
    t_questions, v_questions = dataSplit(questions, imgids, split)
    t_answers, v_answers = dataSplit(answers, imgids, split)
    t_imgids, v_imgids = dataSplit(imgids, imgids, split)

    # Build a dictionary only for training questions.
    worddict, idict = buildDict(t_questions, 1)
    ansdict, iansdict = buildDict(t_answers, 0)
    
    trainInput = combine(\
        lookupQID(t_questions, worddict), t_imgids)
    trainTarget = lookupAnsID(t_answers, ansdict)
    validInput = combine(\
        lookupQID(v_questions, worddict), v_imgids)
    validTarget = lookupAnsID(v_answers, ansdict)

    with open(testQAFilename) as f:
        lines = f.readlines()

    (r_questions, r_answers, r_imgids) = extractQA(lines)
    testInput = combine(\
        lookupQID(r_questions, worddict), r_imgids)
    testTarget = lookupAnsID(r_answers, ansdict)

    worddict_all, idict_all = buildDict(questions, 1)
    ansdict_all, iansdict_all = buildDict(answers, 0)
    allInput = combine(\
        lookupQID(questions, worddict_all), imgids)
    allTarget = lookupAnsID(answers, ansdict_all)

    np.save(\
        os.path.join(outputFolder, 'train-37-unk.npy'),\
        np.array((trainInput, trainTarget, 0),\
            dtype=object))
    np.save(\
        os.path.join(outputFolder, 'valid-37-unk.npy'),\
        np.array((validInput, validTarget, 0),\
            dtype=object))
    np.save(\
        os.path.join(outputFolder, 'test-37-unk.npy'),\
        np.array((testInput, testTarget, 0),\
            dtype=object))
    np.save(\
        os.path.join(outputFolder, 'all-37-unk.npy'),\
        np.array((allInput, allTarget, 0),\
            dtype=object))
    np.save(\
        os.path.join(outputFolder, 'vocab-dict-unk.npy'),\
        np.array((worddict, idict, 
            ansdict, iansdict, 0), dtype=object))
    np.save(\
        os.path.join(outputFolder, 'vocab-dict-unk-all.npy'),\
        np.array((worddict_all, idict_all, 
            ansdict_all, iansdict_all, 0), dtype=object))

    trainInput = combineAttention(\
        lookupQID(t_questions, worddict), t_imgids)
    validInput = combineAttention(\
        lookupQID(v_questions, worddict), v_imgids)
    testInput = combineAttention(\
        lookupQID(r_questions, worddict), r_imgids)
    allInput = combineAttention(\
        lookupQID(questions, worddict_all), imgids)
    np.save(\
        os.path.join(outputFolder, 'train-37-unk-att.npy'),\
        np.array((trainInput, trainTarget, 0),\
            dtype=object))
    np.save(\
        os.path.join(outputFolder, 'valid-37-unk-att.npy'),\
        np.array((validInput, validTarget, 0),\
            dtype=object))
    np.save(\
        os.path.join(outputFolder, 'test-37-unk-att.npy'),\
        np.array((testInput, testTarget, 0),\
            dtype=object))
    np.save(\
        os.path.join(outputFolder, 'all-37-unk-att.npy'),\
        np.array((allInput, allTarget, 0),\
            dtype=object))

    with open(os.path.join(outputFolder, 'question_vocabs.txt'), 'w+') as f:
        for word in idict:
            f.write(word + '\n')

    with open(os.path.join(outputFolder, 'answer_vocabs.txt'), 'w+') as f:
        for word in iansdict:
            f.write(word + '\n')