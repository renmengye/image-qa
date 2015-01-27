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
import tsne

word_array = []

def getTrainData():
    global word_array
    data = np.load('../data/sentiment/train.npy')
    input_ = data[3]
    target_ = data[4]
    word_array = data[1]
    return input_, target_

def getWordEmbedding(initSeed, initRange, pcaDim=0):
    np.random.seed(initSeed)
    weights = np.load('../data/sentiment/vocabs-vec.npy')
    for i in range(weights.shape[0]):
        if weights[i, 0] == 0.0:
            weights[i, :] = np.random.rand(weights.shape[1]) * initRange - initRange / 2.0
    if pcaDim > 0:
        weights = tsne.pca(weights, pcaDim)
    return weights.transpose()

def evaluate(pipeline, input_):
    global word_array
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

    if len(sys.argv) <= 1:
        subset = np.random.permutation(subset)
    trainInput = trainInput[subset]
    trainInput = trainInput.reshape(trainInput.shape[0], trainInput.shape[1], 1)
    trainTarget = trainTarget[subset]
    timespan = trainInput.shape[1]

    if len(sys.argv) > 1:
        with open(sys.argv[1]) as pipf:
            pipeline = pickle.load(pipf)
        evaluate(pipeline, trainInput)
        exit()

    trainOpt = {
        'numEpoch': 2000,
        'needValid': True,
        'heldOutRatio': 0.1,
        'xvalidNo': 0,
        'momentum': 0.9,
        'batchSize': 20,
        'learningRateDecay': 1.0,
        'momentumEnd': 0.9,
        'shuffle': True,
        'writeRecord': True,
        'saveModel': True,
        'plotFigs': True,
        'everyEpoch': True,
        'calcError': True,
        'stopE': 0.01,
        'progress': True,
        'displayDw': 7
    }

    pipeline = Pipeline(
        name='sentiment-vec',
        costFn=crossEntOne,
        decisionFn=hardLimit,
        outputFolder='../results')
    pipeline.addStage(TimeUnfold())
    pipeline.addStage(LinearDict(
        inputDim=np.max(trainInput)+1,
        outputDim=30,
        needInit=False,
        initWeights=getWordEmbedding(
            initSeed=2,
            initRange=0.42,
            pcaDim=30)),       # std ~= 0.12. U~[0.21, 0.21].
        learningRate=0.0)
    pipeline.addStage(TimeFold(
        timespan=timespan))
    pipeline.addStage(Dropout(
        dropoutRate=0.2,
        initSeed=3))
    pipeline.addStage(LSTM(
        inputDim=30,
        outputDim=30,
        initRange=0.1,
        initSeed=4,
        cutOffZeroEnd=True,
        multiErr=True),
        learningRate=1.0,
        annealConst=0.01,
        gradientClip=0.1,
        weightClip=100,
        weightRegConst=5e-5,
        outputdEdX=False)
    pipeline.addStage(Dropout(
        dropoutRate=0.5,
        initSeed=5))
    pipeline.addStage(LSTM(
        inputDim=30,
        outputDim=30,
        initRange=0.1,
        initSeed=6,
        cutOffZeroEnd=True,
        multiErr=False),
        learningRate=0.8,
        annealConst=0.01,
        weightClip=100,
        weightRegConst=5e-5,
        gradientClip=0.1)
    pipeline.addStage(Sigmoid(
        inputDim=30,
        outputDim=1,
        initRange=0.1,
        initSeed=7),
        learningRate=0.01,
        gradientClip=0.1,
        weightClip=1,
        annealConst=0.01,
        weightRegConst=5e-5)
    pipeline.train(trainInput, trainTarget, trainOpt)

