[![PyPI version](https://badge.fury.io/py/keras-video-generators.svg)](https://pypi.org/project/keras-video-generators/) [![Build Status](https://travis-ci.org/metal3d/keras-video-generators.svg?branch=master)](https://travis-ci.org/metal3d/keras-video-generators)

# Keras Sequence Video Generators

This package offers classes that generate sequences of frames from video files using Keras (officially included in TensorFlow as of the 2.0 release). The resulting frame sequences work with the Time Distributed, GRU, LSTM, and other recurrent layers.

See articles:

- [The basics of Video frame as input](https://medium.com/smileinnovation/how-to-work-with-time-distributed-data-in-a-neural-network-b8b39aa4ce00)
- [Usage of these generators here](https://medium.com/smileinnovation/training-neural-network-with-image-sequence-an-example-with-video-as-input-c3407f7a0b0f)

An provided example of usage can be [displayed in nbviewer here](https://nbviewer.jupyter.org/github/metal3d/keras-video-generators/blob/master/Example%20of%20usage.ipynb).

Requirements are:

- Python >= 3.6 (Python 2 will never been supported)
- OpenCV
- numpy
- Keras >= 2
- TensorFlow >= 1.15 (or other backend, not tested, TensorFlow is needed by Keras)

TensorFlow 2 works as well. This requirements **is not integrated in the setup.py** to let you choose the version, or to let you try with other backend. We mean that you will need to install a backend yourself (e.g. `pip install tensorflow`)

If you want to compile the package, you need:

- sphinx to compile doc (work in progress)
- setuptools

# Installation

You can install the package via `pip`:

```bash
pip install keras-video-generators
```

If you want to build from source, clone the repository then:

```bash
python setup.py build
```

# Usage

The module name (keras_video) is different from the installation package name (keras-video-generators). Import the entire module with

```bash
import keras_video
```

or load a single generator:

```bash
from keras_video import VideoFrameGenerator
```

The package contains three generators that inherit the `Sequence` interface and may be used with `model.fit_generator()`:

- `VideoFrameGenerator` provides a pre-determined number of frames from the entire video
- `SlidingFrameGenerator` provides frames with decay for the entire video or with a sequence time
- `OpticalFlowGenerator` provides an optical flow sequence from frames with different methods (experimental)

Each generator accepts a standard set of parameters:

- `glob_pattern`; must contain `{classname}`, e.g. './videos/{classname}/*.avi' - the "classname" in string is used to detect classes
- `nb_frames`; the number of frames in the sequence
- `batch_size`; the number of sequences in one batch
- `transformation`; can be `None` or ImageDataGenerator (Keras) for data augmentation
- `use_frame_cache`; use with caution, if set to `True`, the class will keep frames in memory (without augmentation). You need **a lot of memory**

See the class documentation for all parameters.


# Changelog

## v1.0.14
- Changes to get first and last frames in sequence

## v1.0.13
- try to fix SageMaker problem by avoiding usage of internal keras from tensorflow

## v1.0.12
- fix transformation error with SlidingFrameGenerator

## v1.0.11
- set generator to be Iterable
- frame cache was disabled by error, it's back now
- fixup import Sequence from `tensorflow.keras`
- fix frame count problems for video with bad headers

## v1.0.10
- fix Windows problems with path using backslashes
- add auto discovery for classes if "None" is sent
- add travis tests


## v1.0.9
- fix frame counter in SlidingFrameGenerator

## v1.0.8
- fix tiny video frame count
- refactorisation
- pep8
- fix problems for video without headers

## v1.0.7
- fix name check in classes to get them from filename
- add `split_test` and `split_val`

## v1.0.5
- fix package generation

## v1.0.4 
- fix assertion

