import ml

d = ml.Dataset('/home/danish/Projects/Music-Genre-Classification/DataSet/wav (copy)')

fcd = dict(features=['mfcc', 'mel_spec'], coeffs=dict(mfcc_coeff=20))
fc = ml.FeaturesConfig(**fcd)
print(fc)

rc = ml.ReductionConfig()
print(rc)

train_fe = ml.AudioFeatures(files=d.train_files, features_cfg=fc, reduction_cfg=rc).extract()
test_fe = ml.AudioFeatures(files=d.test_files, features_cfg=fc, reduction_cfg=rc).extract()

ncd = dict(features_dim=train_fe.features_dim, classes=train_fe.classes,
           hidden_units=[300, 350, 550, 900], learn_rate=0.01)
nc = ml.NetworkConfig(**ncd)

net = ml.MLP(nc)
print(net)

net.train(train=train_fe, test=test_fe, epochs=500)

svm = ml.SVM(features_dim=train_fe.features_dim, classes=train_fe.classes, svm_type='nu', kernel='linear', poly_degree=3)
print(svm)

svm.train(train=train_fe, test=test_fe)

