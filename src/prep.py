import os
import numpy as np
import calculate_wups

def buildDict(lines, keystart, pr=False):
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
                word_freq[k - keystart] += 1
    word_dict['UNK'] = key
    word_array.append('UNK')
    sorted_x = sorted(range(len(word_freq)), key=lambda k: word_freq[k], reverse=True)
    if pr:
        summ = 0
        for x in sorted_x:
            print word_array[x], word_freq[x],
            summ += word_freq[x]
        med = summ / 2
        medsumm = 0
        for x in sorted_x:
            if medsumm > med:
                break
            medsumm += word_freq[x]
        print 'median: ', word_array[x], word_freq[x]
        #print sorted_x
        print 'Dictionary length', len(word_dict)
    return  word_dict, word_array, word_freq

def lookupAnsID(answers, ansdict):
    ansids = []
    for ans in answers:
        if ansdict.has_key(ans):
            ansids.append(ansdict[ans])
        else:
            ansids.append(ansdict['UNK'])
    return np.array(ansids, dtype=int).reshape(len(ansids), 1)

def findMaxlen(questions):
    maxlen = 0
    sumlen = 0
    for q in questions:
        words = q.split(' ')
        sumlen += len(words)
        if len(words) > maxlen:
            maxlen = len(words)
    print 'Maxlen:', maxlen
    print 'Mean len:', sumlen / float(len(questions))
    return maxlen

def lookupQID(questions, worddict, maxlen):
    wordslist = []
    for q in questions:
        words = q.split(' ')
        wordslist.append(words)
        if len(words) > maxlen:
            maxlen = len(words)
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
        (np.array(imgids, dtype=int).reshape(len(imgids), 1, 1), \
        wordids), axis=1)

def combineSV(qids, imgids):
    return np.concatenate(
        (np.array(imgids, dtype='int').reshape(len(qids), 1),
        np.array(qids, dtype='int').reshape(len(qids), 1)), axis=1)

def guessBaseline(
                    questions, 
                    answers, 
                    questionTypes):
    """
    Run mode-guessing baseline on a dataset.
    If need to calculate WUPS score, outputFolder must be provided.
    """
    baseline = []
    typedAnswerFreq = []
    for i in range(4):
        typedAnswerFreq.append({})
    for q, a, typ in zip(questions, answers, questionTypes):
        if typedAnswerFreq[typ].has_key(a):
            typedAnswerFreq[typ][a] += 1
        else:
            typedAnswerFreq[typ][a] = 1
    modeAnswers = []
    for typ in range(4):
        tempAnswer = None
        tempFreq = 0
        for k in typedAnswerFreq[typ].iterkeys():
            if typedAnswerFreq[typ][k] > tempFreq:
                tempAnswer = k
                tempFreq = typedAnswerFreq[typ][k]
        modeAnswers.append(tempAnswer)
    for i, ans in enumerate(modeAnswers):
        print 'Baseline answers %d: %s' % (i, ans)

    # Print baseline performance
    baselineCorrect = np.zeros(4)
    baselineTotal = np.zeros(4)
    for n in range(0, len(questions)):
        i = questionTypes[n]
        baseline.append(modeAnswers[i])
        if answers[n] == modeAnswers[i]:
            baselineCorrect[i] += 1
        baselineTotal[i] += 1
    baselineRate = baselineCorrect / baselineTotal.astype('float')
    print 'Baseline rate: %.4f' % \
        (np.sum(baselineCorrect) / np.sum(baselineTotal).astype('float'))
    print 'Baseline object: %.4f' % baselineRate[0]
    print 'Baseline number: %.4f' % baselineRate[1]
    print 'Baseline color: %.4f' % baselineRate[2]
    print 'Baseline location: %.4f' % baselineRate[3]

    # Calculate WUPS score
    wups = np.zeros(3)
    for i, thresh in enumerate([-1, 0.9, 0.0]):
        wups[i] = calculate_wups.runAllList(baseline, answers, thresh)
    print 'Baseline WUPS -1: %.4f' % wups[0]
    print 'Baseline WUPS 0.9: %.4f' % wups[1]
    print 'Baseline WUPS 0.0: %.4f' % wups[2]

    return baseline
