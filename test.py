import os
from itertools import *
import ml
import numpy as np

path = 'DataSet/wav'
all_classes = os.listdir(path)

classes = list(['classical', 'reggae', 'jazz', 'disco'])
dataset = ml.Dataset(path, classes)

train_fe = ml.AudioFeatures(dataset.train_files, vector_reduction='mean')
print(train_fe)

test_fe = ml.AudioFeatures(dataset.test_files, vector_reduction='mean')
print(test_fe)

net = ml.MLP(train_fe.features_dim, train_fe.classes, [280, 300])

print(net)
net.train(train=train_fe, test=test_fe, epochs=500)