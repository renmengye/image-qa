import json
import os
import cPickle as pkl
import sys

if len(sys.argv) < 2:
    tokenFilename = '../../../data/flickr8k/flickr8k.token.txt'
    captionFilename = '../../../data/flickr8k/flickr8k_caption.txt'
    imgidsFilename = '../../../data/flickr8k/flickr8k_imgids.txt'
else:
    tokenFilename = '../../../data/%s/%s.token.txt' % (sys.argv[1],sys.argv[1])
    captionFilename = '../../../data/%s/%s_caption.txt' % (sys.argv[1],sys.argv[1])
    imgidsFilename = '../../../data/%s/%s_imgids.txt' % (sys.argv[1],sys.argv[1])

if __name__ == '__main__':
    with open(tokenFilename) as f:
        captiontxt = f.readlines()
    dataset = []
    for line in captiontxt:
        parts = line.split('\t')
        imgid = parts[0].split('.jpg#')[0]
        sent = parts[1][:-1].replace('\n','').strip()
        dataset.append((sent, imgid))
    with open(captionFilename, 'w') as f:
        for d in dataset:
            f.write('%s\n' % d[0])
    with open(imgidsFilename, 'w') as f:
        for d in dataset:
            f.write('%s\n' % d[1])
