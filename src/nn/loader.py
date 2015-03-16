import yaml
import router
from model2 import *

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
        router.routeStage(stageDict)

    modelStages = []
    for s in modelDict['stages']:
        modelStages.append(router.routeStage(s))
    costFn=router.routeFn(modelDict['costFn'])
    decisionFn=router.routeFn(modelDict['decisionFn'])
    model = GraphModel(
        name=modelDict['name'],
        stages=modelStages,
        costFn=costFn,
        decisionFn=decisionFn,
        specFilename=modelSpecFilename)

    return model