import os
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav

from subprocess import call

def get_mfcc_features(file):
	(rate,sig) = wav.read(file)
	mfcc_feat = mfcc(sig,rate)
	fbank_feat = logfbank(sig,rate)
	print(fbank_feat[1:3,:])


def convert_au_to_wav():
	rootdir = os.getcwd() + '/DataSet/'
	for subdir, dirs, files in os.walk(rootdir):
		call(['mkdir', subdir.replace("au", "wav")])
		for file in files:
			path = os.path.join(subdir, file)
			if(path.split('.')[-1] == 'au'):
				# print(path)
				new_path = path.replace("au", "wav")
				# print(new_path)
				call(['sox', path , '-e', 'signed-integer', new_path])


def loop_through_dataset():
	rootdir = os.getcwd() + '/DataSet/'
	for subdir, dirs, files in os.walk(rootdir):
		for file in files:
			path = os.path.join(subdir, file)
			if(path.split('.')[-1] == 'wav'):
				get_mfcc_features(path)


# convert_au_to_wav()
loop_through_dataset()