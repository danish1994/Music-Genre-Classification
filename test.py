import ml
import numpy as np


for i in range(1,11):
	dataset = ml.Dataset('/home/danish/Projects/Music-Genre-Classification/DataSet/'+str(i))
	print(dataset)
	
	train_fe = ml.AudioFeatures(dataset.train_files, vector_reduction='mean')
	# train_fe.save('./train')
	print(train_fe)
	test_fe = ml.AudioFeatures(dataset.test_files, vector_reduction='mean')
	# test_fe.save('./test')
	print(test_fe)

	# net = ml.MLP(train_fe.features_dim, train_fe.classes, [280, 300, 400, 600, 800])
	net = ml.MLP(train_fe.features_dim, train_fe.classes, [280, 300])

	print(net)
	net.train(train=train_fe, test=test_fe, epochs=500)
