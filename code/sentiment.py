from lstm import *
from simplesum import *
from softmax import *
from sigmoid import *
from time_unfold import *
from time_fold import *
from time_select import *
from linear_map import *
from linear_dict import *
from pipeline import *
from util_func import *
import sys

word_dict = {}
word_array = []

def getTrainData():
    with open('../data/sentiment/train2.txt') as f:
        lines = f.readlines()
        line_max = 0
        sentence_dict = {}
        line_numbers = []
        key = 1
        for i in range(0, len(lines)):
            # Remove duplicate records
            if not sentence_dict.has_key(lines[i]):
                sentence_dict[lines[i]] = 1
                line_numbers.append(i)
                words = lines[i].split(' ')
                for j in range(1, len(words) - 1):
                    if len(words) - 1 > line_max:
                        line_max = len(words) - 1
                    if not word_dict.has_key(words[j]):
                        word_dict[words[j]] = key
                        word_array.append(words[j])
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
            for j in range(1, len(words) - 1):
                input_[count, j - 1] = word_dict[words[j]]
                #input_[count, j - 1, word_dict[words[j]]] = 1.0
            count += 1
    return input_, target_

def getTestData():
    input_ = 0
    return input_

if __name__ == '__main__':
    trainInput, trainTarget = getTrainData()          # 2250 records

    np.random.seed(1)
    subset = np.arange(0, 129 * 2)                    # 522 records, 129 positive
    if len(sys.argv) < 3:
        subset = np.random.permutation(subset)
    trainInput = trainInput[subset]
    trainInput = trainInput.reshape(trainInput.shape[0], trainInput.shape[1], 1)
    trainTarget = trainTarget[subset]
    timespan = trainInput.shape[1]

    trainOpt = {
        'numEpoch': 2000,
        'heldOutRatio': 0.1,
        'momentum': 0.9,
        'batchSize': 1,
        'learningRateDecay': 1.0,
        'momentumEnd': 0.9,
        'shuffle': True,
        'needValid': True,
        'writeRecord': True,
        'plotFigs': True,
        'everyEpoch': True,
        'calcError': True,
        'stopE': 0.01
    }

    if len(sys.argv) <= 1:
        pipeline = Pipeline(
            name='sentiment',
            costFn=crossEntOne,
            decisionFn=hardLimit)
        pipeline.addStage(TimeUnfold())
        pipeline.addStage(LinearDict(
            inputDim=np.max(trainInput)+1,
            outputDim=20,
            initRange=0.001,
            initSeed=2),
            learningRate=.1)
        pipeline.addStage(TimeFold(
            timespan=timespan))
        pipeline.addStage(LSTM(
            inputDim=20,
            memoryDim=10,
            initRange=0.01,
            initSeed=3,
            cutOffZeroEnd=True,
            dropoutRate=0.5),
            learningRate=.1)
        pipeline.addStage(LSTM(
            inputDim=10,
            memoryDim=10,
            initRange=0.01,
            initSeed=3,
            cutOffZeroEnd=True,
            dropoutRate=0.5),
            learningRate=.05)
        pipeline.addStage(TimeSelect(
            time=-1))
        pipeline.addStage(Sigmoid(
            inputDim=10,
            outputDim=1,
            initRange=0.01,
            initSeed=4),
            learningRate=.005)

    if len(sys.argv) > 1:
        with open(sys.argv[1] + '.pip') as pipf:
            pipeline = pickle.load(pipf)
        if len(sys.argv) > 2:
            if sys.argv[2] == '-test':
                trainOutput = pipeline.forwardPass(trainInput.transpose((1, 0, 2)))
                with open('sentiment_result.txt', 'w+') as f:
                    for n in range(trainOutput.shape[0]):
                        sentence = '%d ' % argmax(trainOutput[n])
                        for t in range(trainInput[n].shape[0]):
                            if trainInput[n, t] == 0 or trainInput[n, t] == '\n':
                                break
                            sentence += word_array[trainInput[n, t] - 1] + ' '
                        f.write(sentence + '\n')
        else:
            pipeline.train(trainInput, trainTarget, trainOpt)
    else:
        pipeline.train(trainInput, trainTarget, trainOpt)


    #pipeline.save()
    pass
