import json
import os
import cPickle as pkl
import sys

if len(sys.argv) < 2:
    folder = 'train'
else:
    folder = sys.argv[1]

if folder == 'train':
    filename = 'train'
elif folder == 'valid':
    filename = 'val'
    
jsonFilename = '/ais/gobi3/datasets/mscoco/annotations/captions_%s2014.json' % filename
outputFolder = '/ais/gobi3/u/mren/data/mscoco/%s' % folder
captionOut = os.path.join(outputFolder, 'captions.txt')
imgidOut = os.path.join(outputFolder, 'imgids.txt')

# To retrive image ID and url, get caption['images'][i]['id'] and caption['images'][i]['url']
if __name__ == '__main__':
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    with open(jsonFilename) as f:
        captiontxt = f.read()
    caption = json.loads(captiontxt)
    L = len(caption['annotations'])

    dataset = []
    for i in range(L):
        sent = caption['annotations'][i]['caption']
        imgid = caption['annotations'][i]['image_id']
        sent = sent.replace('\n','').strip().strip('.').strip()
        dataset.append((sent, imgid))
        #if sent.startswith('One motorcycle witha large'):
        #print (sent, imgid)

    with open(captionOut, 'w') as f:
        for d in dataset:
            f.write(d[0] + '\n')
    with open(imgidOut, 'w') as f:
        for d in dataset:
            f.write(str(d[1]) + '\n')
    # with open('../../../data/mscoco/mscoco_caption.pkl', 'wb') as f:
    #     pkl.dump(dataset, f)
