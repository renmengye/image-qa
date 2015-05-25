import sys
import os
import shutil
import imageqa_test as it
import numpy as np

realValueDict = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'eleven': 11,
    'twelve': 12
}

realValueIdict = [
    'zero',
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'eight',
    'nine',
    'ten',
    'eleven',
    'twelve'
]

if __name__ == '__main__':
    identity = False
    for i, flag in enumerate(sys.argv):
        if flag == '-d' or flag == '-data':
            numDataFolder = sys.argv[i + 1]
        elif flag == '-o' or flag == '-output':
            outputFolder = sys.argv[i + 1]
        elif flag == '-i' or flag == '-identity':
            identity = True
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    data = it.loadDataset(numDataFolder)
    targetDataRV = []
    inputDataRV = []
    ansIdict = data['ansIdict']
    eye = np.eye(len(ansIdict))

    for sepData in ['trainData', 'validData', 'testData']:
        inputData = data[sepData][0]
        targetData = data[sepData][1]
        targetDataRVSep = []
        inputDataRVSep = []

        for n in range(targetData.shape[0]):
            print targetData[n, 0]
            ans = ansIdict[targetData[n, 0]]
            print ans
            if realValueDict.has_key(ans):
                value = realValueDict[ansIdict[targetData[n, 0]]]
                if identity:
                    targetDataRVSep.append(eye[value])
                else:
                    targetDataRVSep.append(value)
                        
                inputDataRVSep.append(inputData[n, :, 0])

        targetDataRVSep = np.array(targetDataRVSep)
        inputDataRVSep = np.array(inputDataRVSep)

        if identity:
            targetDataRVSep = \
                targetDataRVSep.reshape(targetDataRVSep.shape[0], len(ansIdict))
        else:
            targetDataRVSep = \
                targetDataRVSep.reshape(targetDataRVSep.shape[0], 1)
        inputDataRVSep = \
            inputDataRVSep.reshape(inputDataRVSep.shape[0], inputDataRVSep.shape[1], 1)
        print targetDataRVSep.shape, inputDataRVSep.shape

        targetDataRV.append(targetDataRVSep)
        inputDataRV.append(inputDataRVSep)

    for d in targetDataRV:
        print d.shape

    for d in inputDataRV:
        print d.shape

    np.save(os.path.join(outputFolder, 'train.npy'),
        np.array(
            (inputDataRV[0], targetDataRV[0], 0), dtype='object'))
    np.save(os.path.join(outputFolder, 'valid.npy'),
        np.array(
            (inputDataRV[1], targetDataRV[1], 0), dtype='object'))
    np.save(os.path.join(outputFolder, 'test.npy'),
        np.array(
            (inputDataRV[2], targetDataRV[2], 0), dtype='object'))

    shutil.copyfile(
        os.path.join(numDataFolder, 'vocab-dict.npy')
        os.path.join(outputFolder, 'vocab-dict.npy'))