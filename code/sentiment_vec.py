from lstm import *
from simplesum import *
from softmax import *
from sigmoid import *
from time_unfold import *
from time_fold import *
from time_select import *
from dropout import *
from linear_map import *
from linear_dict import *
from pipeline import *
from util_func import *
import sys

word_array = []

def getTrainData():
    global word_array
    data = np.load('../data/sentiment/train.npy')
    input_ = data[3]
    target_ = data[4]
    word_array = data[1]
    return input_, target_

def getWordEmbedding(initSeed, initRange):
    np.random.seed(initSeed)
    weights = np.load('../data/sentiment/vocabs-vec.npy')
    for i in range(weights.shape[0]):
        if weights[i, 0] == 0.0:
            weights[i, :] = np.random.rand(weights.shape[1]) * initRange - initRange / 2.0
    return weights

def evaluate(pipelineFilename, input_):
    global word_array
    with open(pipelineFilename) as pipf:
        pipeline = pickle.load(pipf)
        output = pipeline.forwardPass(input_.transpose((1, 0, 2)), dropout=False)
        with open('../results/sentiment_result.txt', 'w+') as f:
            for n in range(output.shape[0]):
                sentence = '%d ' % hardLimit(output[n])
                for t in range(input_[n].shape[0]):
                    if input_[n, t] == 0 or input_[n, t] == '\n':
                        break
                    sentence += word_array[trainInput[n, t] - 1] + ' '
                f.write(sentence + '\n')

if __name__ == '__main__':
    trainInput, trainTarget = getTrainData()          # 2250 records
    np.random.seed(1)
    subset = np.arange(0, 129 * 2)                    # 522 records, 129 positive
    subset = np.random.permutation(subset)
    trainInput = trainInput[subset]
    trainInput = trainInput.reshape(trainInput.shape[0], trainInput.shape[1], 1)
    trainTarget = trainTarget[subset]
    timespan = trainInput.shape[1]

    if len(sys.argv) > 1:
        with open(sys.argv[1]) as pipf:
            pipeline = pickle.load(pipf)
        evaluate(pipeline, trainInput, trainTarget)
        exit()

    trainOpt = {
        'numEpoch': 2000,
        'heldOutRatio': 0.1,
        'momentum': 0.9,
        'batchSize': 20,
        'learningRateDecay': 1.0,
        'momentumEnd': 0.9,
        'shuffle': True,
        'needValid': True,
        'writeRecord': True,
        'saveModel': True,
        'plotFigs': True,
        'everyEpoch': True,
        'calcError': True,
        'stopE': 0.01,
        'progress': True,
        'displayDw': 4
    }

    pipeline = Pipeline(
        name='sentiment',
        costFn=crossEntOne,
        decisionFn=hardLimit,
        outputFolder='../results')
    pipeline.addStage(TimeUnfold())
    pipeline.addStage(LinearDict(
        inputDim=np.max(trainInput)+1,
        outputDim=300,
        W=getWordEmbedding(initSeed=2, initRange=0.42)),       # std ~= 0.12. U~[0.21, 0.21].
        learningRate=0.0)
    pipeline.addStage(TimeFold(
        timespan=timespan))
    pipeline.addStage(Dropout(
        dropoutRate=0.2))
    pipeline.addStage(LSTM(
        inputDim=300,
        memoryDim=100,
        initRange=0.1,
        initSeed=3,
        cutOffZeroEnd=True),
        learningRate=0.8,
        weightClip=0.1)
    pipeline.addStage(Dropout(
        dropoutRate=0.5))
    pipeline.addStage(LSTM(
        inputDim=100,
        memoryDim=50,
        initRange=0.1,
        initSeed=4,
        cutOffZeroEnd=True),
        learningRate=0.8,
        weightClip=0.1)
    pipeline.addStage(TimeSelect(
        time=-1))
    pipeline.addStage(Sigmoid(
        inputDim=50,
        outputDim=1,
        initRange=0.1,
        initSeed=5),
        learningRate=0.01)
    pipeline.train(trainInput, trainTarget, trainOpt)

