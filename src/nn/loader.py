import yaml
import router

def load(configFilename):
    with open(configFilename) as f:
        modelDict = yaml.load(f)

    for stageDict in modelDict['stages']:
        router.routeStage(stageDict)

    costFn=router.routeFn(modelDict['costFn'])
    decisionFn=router.routeFn(modelDict['decisionFn'])
    model = router.getStage(modelDict['model'])
    model.getCost = costFn
    model.predict = decisionFn
    model.configFilename = configFilename

    return model