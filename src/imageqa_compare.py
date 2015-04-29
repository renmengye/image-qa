import sys
import os
from imageqa_test import *
import nn
import numpy as np

if __name__ == '__main__':
    """
    Usage python imageqa_compare.py -m[odels] {id1},{id2},{id3}...
                                    -d[ata] {dataFolder}
                                    -o[utput] {outputFolder}
    """
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '-m' or sys.argv[i] == '-models':
            modelsStr = sys.argv[i + 1]
            modelIds = modelsStr.split(',')
        elif sys.argv[i] == '-d' or sys.argv[i] == '-data':
            dataFolder = sys.argv[i + 1]
        elif sys.argv[i] == '-o' or sys.argv[i] == '-output':
            outputFolder = sys.argv[i + 1]
    resultsFolder = '../results'
    K = 3 # Top-K answers
    modelOutputs = []

    print 'Loading test data...'
    testDataFile = os.path.join(dataFolder, 'test.npy')
    testData = np.load(testDataFile)
    inputTest = testData[0]
    targetTest = testData[1]
    questionArray = vocabDict[1]
    answerArray = vocabDict[3]
    testQuestionTypeFile = os.path.join(dataFolder, 'test-qtype.npy')
    testQuestionType = np.load(testQuestionTypeFile)

    print 'Loading model comparison files...'
    for modelId in modelIds:
        print modelId
        compareFile = '%s/%s/%s.test.comp.txt' % \
            (resultsFolder, modelId, modelId)
        if os.path.exists(compareFile):
            with open(compareFile) as f:
                modelOutputs.append(f.readlines())
        else:
            print 'Running test data on model %s' \
                    % modelId
            testTruthFile = os.path.join(
                            resultsFolder, taskId, 
                            '%s.test.t.txt' % taskId)
            outputTest = nn.test(model, inputTest)
            outputTxt(outputTest, targetTest, 
                answerArray, compareFile, 
                testTruthFile, topK=K, outputProb=True):
            modelOutputs.append(f.readlines())

    # Sort questions by question types.
    categoryInput = np.zeros(4, dtype='object')


    # Sort questions by correctness differences.