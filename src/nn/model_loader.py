import yaml
import layer_factory
from graph_model import *

def load(modelSpecFilename):
    """
    Need the following items in the model spec file:
    costFn
    decisionFn
    stages
    specs
    :param modelSpecFilename:
    :return:
    """
    with open(modelSpecFilename) as f:
        modelDict = yaml.load(f)

    for stageDict in modelDict['specs']:
        layer_factory.addStage(stageDict)

    modelStages = []
    for s in modelDict['stages']:
        modelStages.append(layer_factory.routeStage(s))
    costFn=layer_factory.routeFn(modelDict['costFn'])

    if modelDict.has_key('decisionFn'):
        decisionFn = layer_factory.routeFn(modelDict['decisionFn'])
    else:
        decisionFn = None
    outputList = modelDict['outputs'].split(',')
    for i in range(len(outputList)):
        outputList[i] = outputList[i].strip()
    model = GraphModel(
        name=modelDict['name'] if modelDict.has_key('name') else None,
        stages=modelStages,
        outputStageNames=outputList,
        costFn=costFn,
        decisionFn=decisionFn,
        specFilename=modelSpecFilename)

    return model