# Keras Sequence Video generators

This package proposes some classes to work with Keras (included in TensorFlow) that generates batches of frames from video files.

It is useful to work with Time Distributed Layer and GRU/LSTM.

Requirements are:

- Python3 (Python 2 will never been supported)
- OpenCV
- numpy
- Keras
- TensorFlow (or other backend, not tested)

If you want to compile the package, you need:

- sphinx to compile doc
- setuptools

# Installation

You can install the package via `pip`:

```bash
pip install keras-video-generators
```

If you want to build from sources, clone the repository then:

```bash
python setup.py build
```

# Usage

The package contains 3 generators that inherits `Sequence` interface. So they may be used with `model.fit_generator()`:

- `VideoFrameGenerator` that will take the choosen number of frames from the entire video
- `SlidingFrameGenerator` that takes frames with decay for the entire video or with a sequence time
- `OpticalFlowGenerator` that gives optical flow sequence from frames with different methods

Each of these generators accepts parameters:

- `glob_pattern` that must contain `{classname}`, e.g. './videos/{classname}/*.png' - the "classname" in string is used to detect classes
- `nb_frame` that is the number of frame in the sequence
- `batch_size`
- `transformation` that can be `None` or or ImageDataGenerator to make data augmentation
- `use_frame_cache` to use with caution, if set to `True`, the class will keep frames in memory (without augmentation). You need **a lot of memory**
- and many more, see class documentation


# Work in progress

- [ ] Complete documentation
- [ ] use Optical Flow on SlidingFrameGenerator
