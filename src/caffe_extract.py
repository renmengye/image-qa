import numpy as np
import caffe
import skimage.transform

MODEL_FILE = \
    '/ais/gobi3/u/mren/models/places/hybridCNN_deploy_upgraded.prototxt'
PRETRAINED = \
    '/ais/gobi3/u/mren/models/places/hybridCNN_iter_700000_upgraded.caffemodel'
IMAGE_FILE = '/u/mren/data/nyu-depth/jpg/image%d.jpg'
IMAGE_MEAN = \
    '/ais/gobi3/u/mren/models/places/hybridCNN_mean.binaryproto'

CATEGORY_FILE = '/ais/gobi3/u/mren/models/places/categoryIndex_hybridCNN.csv'
SYNSETS_FILE = '/ais/gobi3/u/mren/models/places/synsets.txt'

caffe.set_mode_gpu()

meanblob = caffe.proto.caffe_pb2.BlobProto()
data = open(IMAGE_MEAN, 'rb').read()
meanblob.ParseFromString(data)
meanarr = np.array(caffe.io.blobproto_to_array(meanblob))
meanarr = meanarr.reshape(3, 256, 256)

net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', meanarr.mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

net.blobs['data'].reshape(1,3,227,227)
features = np.zeros((1449, 4096), dtype='float32')

cls_dict = {}
synsets_dict = {}
with open(SYNSETS_FILE) as f:
    for line in f:
        parts = line.split(' ')
        synsets_dict[parts[0]] = line
with open(CATEGORY_FILE) as f:
    for line in f:
        parts = line.split(' ')
        if synsets_dict.has_key(parts[0]):
            cls_dict[int(parts[1])] = synsets_dict[parts[0])
        else:
            cls_dict[int(parts[1])] = parts[0]

for i in range(1, 1450):
    print i
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(IMAGE_FILE % i))
    out = net.forward()
    pred_cls = out['prob'].argmax()
    print("Predicted class is #{}.".format(pred_cls)),
    print cls_dict[pred_cls]
    features[i-1] = net.blobs['fc7'].data[0]

np.save('/ais/gobi3/u/mren/data/nyu-depth/hidden7_places.npy', features)
