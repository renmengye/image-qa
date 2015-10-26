import imageqa_test
import h5py
import sys

if __name__ == '__main__':
    """
    Test the first 100 entries and output the activation for each layer.
    Usage python imageqa_prob.py -m[odel] {Model ID} 
                                 -d[ata] {dataFolder}
                                 -o[utput] {output H5 file}
                                 [-r[esults] {resultsFolder}]
    """
    dataFolder = None
    resultsFolder = '../results'
    for i, flag in enumerate(sys.argv):
        if flag == '-m' or flag == '-model':
            modelId = sys.argv[i + 1]
        elif flag == '-d' or flag == '-data':
            dataFolder = sys.argv[i + 1]
        elif flag == '-o' or flag == '-output':
            outputFile = sys.argv[i + 1]
        elif flag == '-r' or flag == '-result':
            resultsFolder = sys.argv[i + 1]
    print modelId
    model = imageqa_test.loadModel(modelId, resultsFolder)
    data = imageqa_test.loadDataset(dataFolder)
    model.forward(data['testData'][0][:100])
    output = h5py.File(outputFile, 'w')
    for stage in model.stages:
        print stage.name
    output['txtDict'] = model.stageDict['txtDict'].Y
    output['bow'] = model.stageDict['bow'].Y
    output['imgFeat'] = model.stageDict['imgFeat'].Y
    output['answer'] = model.stageDict['softmax'].Y
    output.close()

