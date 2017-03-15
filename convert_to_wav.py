import os
import scipy.io.wavfile as wav

from subprocess import call

def convert_au_to_wav():
	rootdir = os.getcwd() + 'DataSet/'
	print(rootdir)
	for subdir, dirs, files in os.walk(rootdir):
		call(['mkdir', subdir.replace("au", "wav")])
		for file in files:
			path = os.path.join(subdir, file)
			if(path.split('.')[-1] == 'au'):
				# print(path)
				new_path = path.replace("au", "wav")
				# print(new_path)
				call(['sox', path , '-e', 'signed-integer', new_path])


convert_au_to_wav()
