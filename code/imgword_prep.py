import numpy as np
import re

def escapeNumber(line):
    line = re.sub('11', 'eleven', line)
    line = re.sub('12', 'twelve', line)
    line = re.sub('13', 'thirteen', line)
    line = re.sub('14', 'fourteen', line)
    line = re.sub('15', 'fifteen', line)
    line = re.sub('16', 'sixteen', line)
    line = re.sub('17', 'seventeen', line)
    line = re.sub('18', 'eighteen', line)
    line = re.sub('19', 'nineteen', line)
    line = re.sub('20', 'twenty', line)
    line = re.sub('1', 'one', line)
    line = re.sub('2', 'two', line)
    line = re.sub('3', 'three', line)
    line = re.sub('4', 'four', line)
    line = re.sub('5', 'five', line)
    line = re.sub('6', 'six', line)
    line = re.sub('7', 'seven', line)
    line = re.sub('8', 'eight', line)
    line = re.sub('9', 'nine', line)
    line = re.sub('0', 'zero', line)

    return line

def buildDict(lines, lowFreq):
    # From word to number.
    word_dict = {}
    # From number to word, numbers need to minus one to convert to list indices.
    word_array = []
    # Word frequency
    word_freq = []
    # Key is 1-based, 0 is reserved for sentence end.
    key = 1
    for i in range(0, len(lines)):
        words = lines[i].split(' ')
        for j in range(0, len(words)):
            if not word_dict.has_key(words[j]):
                word_dict[words[j]] = key
                word_array.append(words[j])
                word_freq.append(1)
                key += 1
            else:
                k = word_dict[words[j]]
                word_freq[k - 1] += 1

    # Sort frequency
    sorted_key = sorted(range(len(word_freq)), key=lambda k: word_freq[k], reverse=True)

    # Replace low frequency words as unk_
    key = 1
    word_dict_unk = {}
    word_array_unk = []
    lowest_freq = lowFreq
    unknown = 'unk_'
    count = 0
    for k in sorted_key:
        count += 1
        if word_freq[k] < lowest_freq:
            break
    word_dict_unk[unknown] = count
    for i in range(0, len(lines)):
        words = lines[i].split(' ')
        for j in range(0, len(words)):
            word = words[j]
            if not word_dict_unk.has_key(word):
                if word_freq[word_dict[word] - 1] >= lowest_freq:
                    word_dict_unk[word] = key
                    key += 1
                    word_array_unk.append(word)
    word_array_unk.append(unknown)

    return  word_dict_unk, word_array_unk

def buildInputTarget(lines, numEx, lineMax, wordDict):
    input_ = np.zeros((numEx, lineMax, 1), dtype=int)
    target_ = np.zeros((numEx, 1), dtype=int)
    for i in range(0, numEx):
        n = i * 2
        words = lines[n].split(' ')
        # Ignore two word answers for now.
        if wordDict.has_key(lines[n + 1]):
            target_[i, 0] = wordDict[lines[n + 1]]
            for t in range(0, len(words)):
                input_[i, t, 0] = wordDict[words[t]]
    return input_, target_

if __name__ == '__main__':
    # Image ID
    imgIds = []

    with open('../data/mpi-qa/qa.37.raw.train.txt') as f:
        lines = f.readlines()

    numTrain = len(lines) / 2
    with open('../data/mpi-qa/qa.37.raw.test.txt') as f:
        lines.extend(f.readlines())
    numTest = len(lines) / 2 - numTrain

    imgIdPattern = re.compile('image')
    pureQA = []
    lineMax = 0
    for i in range(0, len(lines) / 2):
        n = i * 2
        match = re.search('image(\d+)', lines[n])
        number = int((re.search('\d+', match.group())).group())
        pureQA.append(re.sub(' in the image(\d+) \?\s', '' ,lines[n]))
        pureQA.append(escapeNumber(re.sub('\s$', '', lines[n + 1])))
        imgIds.append(number)
        if len(pureQA[n]) > lineMax: lineMax = len(pureQA[n])

    wordDict, wordArray = buildDict(pureQA, 1)

    trainInput, trainTarget = buildInputTarget(pureQA[:numTrain*2], numTrain, lineMax, wordDict)
    trainInput = np.concatenate((np.reshape(imgIds[:numTrain], (numTrain, 1, 1)), trainInput), axis=1)
    testInput, testTarget = buildInputTarget(pureQA[numTrain*2:], numTest, lineMax, wordDict)
    testInput = np.concatenate((np.reshape(imgIds[numTrain:], (numTest, 1, 1)), testInput), axis=1)
    data = np.array((trainInput, trainTarget, testInput, testTarget, 0), dtype=object)
    np.save('../data/imgword/train-37.npy', data)


