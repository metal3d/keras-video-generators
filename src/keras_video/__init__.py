"""
Video Generator package
=======================

Provides generators for video sequence that can be injected in Time Distributed layer.

It is made to provide several types of sequences:

- Frames from video
- Sliding samples of frame from video
- Optical Flow variation

So the package provides 3 classes. The mother class is VideoFrameGenerator,
and the other ones inherits from it.

Each generator can use ImageDataGenerator from keras to make data augmentation,
and takes common params as `nb_frames`, `batch_size`, and so on.

The goal is to provide ``(BS, N, W, H, C)`` shape of data where:

- ``BS`` is the batch size
- ``N`` is the number of frames
- ``W`` and ``H`` are width and height
- ``C`` is the number of channels (1 for gray scale, 3 for RGB)

For example ``(16, 5, 224, 224, 3)`` is a batch of 16 sequences where
sequence has got 5 frames sized to ``(224, 224)`` in RGB.

"""

__version__ = "1.0.13"

from . import flow
from . import generator
from . import sliding

VideoFrameGenerator = generator.VideoFrameGenerator
OpticalFlowGenerator = flow.OpticalFlowGenerator
SlidingFrameGenerator = sliding.SlidingFrameGenerator

METHOD_OPTICAL_FLOW = flow.METHOD_OPTICAL_FLOW
METHOD_FLOW_MASK = flow.METHOD_FLOW_MASK
METHOD_DIFF_MASK = flow.METHOD_DIFF_MASK
METHOD_ABS_DIFF = flow.METHOD_ABS_DIFF
