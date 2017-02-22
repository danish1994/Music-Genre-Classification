from setuptools import setup

setup(name='Music Genre Classification Classification',
      version='0.1',
      description='Music Genre Classification in Python',
      url='https://github.com/danish1994/Music-Genre-Classification',
      author='Danish',
      author_email='danish8802204230@gmail.com',
      license='MIT',
      packages=['src'],
      install_requires=[
          'numpy',
          'scipy',
          'scikit-learn',
          'matplotlib',
          'setuptools',
          'python-dateutil',
          'pyparsing',
          'pytz',
          'cycler',
          'lda',
          'pypdf2',
          'tqdm',
          'python_speech_features'
      ])
