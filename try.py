import os
from itertools import *
import ml
import numpy as np

path = '/home/danish/Projects/Music-Genre-Classification/DataSet/wav'
all_classes = os.listdir(path)

for i in reversed(range(2,7)):
	classes_lists = list(combinations(all_classes, i))
	for classes_list in classes_lists:

		classes = list(classes_list)
		dataset = ml.Dataset(path, classes)
		
		train_fe = ml.AudioFeatures(dataset.train_files, vector_reduction='mean')
		print(train_fe)
	
		test_fe = ml.AudioFeatures(dataset.test_files, vector_reduction='mean')
		print(test_fe)

		net = ml.MLP(train_fe.features_dim, train_fe.classes, [280, 300])

		print(net)
		net.train(train=train_fe, test=test_fe, epochs=500)