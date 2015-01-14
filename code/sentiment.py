from lstm import *
from simplesum import *
from softmax import *
from time_unfold import *
from time_fold import *
from time_select import *
from linear_map import *
from linear_dict import *
from pipeline import *
from util_func import *

def getTrainData():
    with open('../data/sentiment/train2.txt') as f:
        lines = f.readlines()
        line_max = 0
        word_dict = {}
        sentence_dict = {}
        line_numbers = []
        key = 1
        for i in range(0, len(lines)):
            # Remove duplicate records
            if not sentence_dict.has_key(lines[i]):
                sentence_dict[lines[i]] = 1
                line_numbers.append(i)
                words = lines[i].split(' ')
                for j in range(1, len(words)):
                    if len(words) - 1 > line_max:
                        line_max = len(words) - 1
                    if not word_dict.has_key(words[j]):
                        word_dict[words[j]] = key
                        key += 1

        #input_ = np.zeros((len(line_numbers), line_max, len(word_dict)), float)
        input_ = np.zeros((len(line_numbers), line_max), int)
        target_ = np.zeros((len(line_numbers), 1), int)
        count = 0
        for i in line_numbers:
            if lines[i][0] == 'p':
                target_[count, 0] = 1
            else:
                target_[count, 0] = 0
            words = lines[i].split(' ')
            for j in range(1, len(words)):
                input_[count, j - 1] = word_dict[words[j]]
                #input_[count, j - 1, word_dict[words[j]]] = 1.0
            count += 1
    return input_, target_

def getTestData():
    input_ = 0
    return input_

if __name__ == '__main__':
    trainInput, trainTarget = getTrainData()          # 2250 records
    #testInput = getTestData()                                   # 11548 records
    #subset = np.arange(0, 129 * 2)
    subset = np.arange(0, 522)
    subset = np.random.permutation(subset)
    #subset = np.random.permutation(subset)[0:20]
    trainInput = trainInput[subset]
    trainInput = trainInput.reshape(trainInput.shape[0], trainInput.shape[1], 1)
    trainTarget = trainTarget[subset]
    timespan = trainInput.shape[1]
    pipeline = Pipeline(
        name='sentiment',
        costFn=crossEntIdx,
        decisionFn=argmax)
    pipeline.addStage(TimeUnfold())
    pipeline.addStage(LinearDict(
        inputDim=np.max(trainInput)+1,
        outputDim=20,
        initRange=1,
        initSeed=2))
    pipeline.addStage(TimeFold(
        timespan=timespan))
    pipeline.addStage(LSTM(
        inputDim=20,
        memoryDim=10,
        initRange=0.01,
        initSeed=3))
    pipeline.addStage(TimeSelect(
        time=-1))
    pipeline.addStage(Softmax(
        inputDim=10,
        outputDim=2,
        initRange=1,
        initSeed=4))
    trainOpt = {
        'learningRate': 100.0,
        'numEpoch': 2000,
        'heldOutRatio': 0.8,
        'momentum': 0.9,
        'batchSize': 1,
        'learningRateDecay': 1.0,
        'momentumEnd': 0.9,
        'needValid': True,
        'plotFigs': True,
        'calcError': True,
        'stopE': 0.01
    }

    pipeline.train(trainInput, trainTarget, trainOpt)
    #testOutput = pipeline.forwardPass(testInput)
    pipeline.save()
    pass
