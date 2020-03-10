"""
Optical Flow Generator
----------------------

.. warning::
    This module is not stable !

The purpose of that module is to return optical flow sequences from a video.

Several methods are defined:

    - Use standard optical flow METHOD_OPTICAL_FLOW=1
    - Use optical flow as a mask on video METHOD_FLOW_MASK=2
    - Use absolute diff mask on video METHOD_DIFF_MASK=3
    - Use abs diff  METHOD_ABS_DIFF=4

"""

import numpy as np
import cv2 as cv
from .generator import VideoFrameGenerator
import keras.preprocessing.image as kimage

METHOD_OPTICAL_FLOW = 1
METHOD_FLOW_MASK = 2
METHOD_DIFF_MASK = 3
METHOD_ABS_DIFF = 4


class OpticalFlowGenerator(VideoFrameGenerator):
    """ Generate optical flow sequence from frames in videos. It can
    use different methods.

    params:

    - method: METHOD_OPTICAL_FLOW, METHOD_FLOW_MASK, METHOD_DIFF_MASK, \
        METHOD_ABS_DIFF
    - flowlevel: integer that give the flow level to calcOpticalFlowFarneback
    - iterations: integer number of iterations for calcOpticalFlowFarneback
    - winsize: flow window size for calcOpticalFlowFarneback

    from VideoFrameGenerator:

    - rescale: float fraction to rescale pixel data (commonly 1/255.)
    - nb_frames: int, number of frames to return for each sequence
    - classes: list of str, classes to infer
    - batch_size: int, batch size for each loop
    - use_frame_cache: bool, use frame cache (may take a lot of memory for \
        large dataset)
    - shape: tuple, target size of the frames
    - shuffle: bool, randomize files
    - transformation: ImageDataGenerator with transformations
    - split: float, factor to split files and validation
    - nb_channel: int, 1 or 3, to get grayscaled or RGB images
    - glob_pattern: string, directory path with '{classname}' inside that
        will be replaced by one of the class list
    """

    def __init__(
            self,
            *args,
            nb_frames=5,
            method=METHOD_OPTICAL_FLOW,
            flowlevel=3,
            iterations=3,
            winsize=15,
            **kwargs):
        super().__init__(nb_frames=nb_frames+1, *args, **kwargs)
        self.flowlevel = flowlevel
        self.iterations = iterations
        self.winsize = winsize
        self.method = method

    def absdiff(self, images):
        """ Get absolute differences between 2 images """

        assert len(images) == 2

        images = list(images)

        for i, image in enumerate(images):
            if image.shape[2] == 3:
                images[i] = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

        diff = cv.absdiff(images[0], images[1])

        if self.nb_channel == 3:
            diff = cv.cvtColor(diff, cv.COLOR_GRAY2RGB)

        return diff

    def make_optical_flow(self, images):
        """ Process Farneback Optical Flow on images"""

        assert len(images) == 2

        images = list(images)
        model = images[0]
        if len(model.shape) == 3 and model.shape[2] == 1:
            model = cv.cvtColor(model, cv.COLOR_GRAY2RGB)

        hsv = np.zeros_like(model)
        hsv[..., 1] = 255

        for i, image in enumerate(images):
            if image.shape[2] == 3:
                images[i] = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

        flow = cv.calcOpticalFlowFarneback(
            images[0], images[1],  # image prev and next
            None, 0.5, self.flowlevel,
            self.winsize, self.iterations,
            5, 1.1, 0)

        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        if self.nb_channel == 1:
            rgb = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)

        return rgb

    def diff_mask(self, images):
        """ Get absolute diff mask, then merge frames and apply the mask
        """
        mask = self.absdiff(images)
        mask = cv.GaussianBlur(mask, (15, 15), 0)

        image = cv.addWeighted(images[0], .5, images[1], .5, 0)

        return cv.multiply(image, mask)

    def flow_mask(self, images):
        """
        Get optical flow on images, then merge images and apply the mask
        """
        mask = self.make_optical_flow(images) / 255.
        mask = cv.GaussianBlur(mask, (15, 15), 0)

        image = cv.addWeighted(images[0], .5, images[1], .5, 0)

        return cv.multiply(image, mask)

    def get_validation_generator(self):
        """ Return the validation generator if you've provided split factor """
        return self.__class__(
            method=self.method,
            nb_frames=self.nbframe-1,
            nb_channel=self.nb_channel,
            target_shape=self.target_shape,
            classes=self.classes,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            rescale=self.rescale,
            glob_pattern=self.glob_pattern,
            _validation_data=self.validation)

    def get_test_generator(self):
        """ Return the validation generator if you've provided split factor """
        return self.__class__(
            method=self.method,
            nb_frames=self.nbframe-1,
            nb_channel=self.nb_channel,
            target_shape=self.target_shape,
            classes=self.classes,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            rescale=self.rescale,
            glob_pattern=self.glob_pattern,
            _test_data=self.test)

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        images = []

        for item in batch[0]:
            imgs = item
            batch_len = len(imgs)
            frames = []
            for i in range(batch_len-1):
                im1 = imgs[i]
                im2 = imgs[i+1]
                if self.method == METHOD_OPTICAL_FLOW:
                    image = self.make_optical_flow((im1, im2))
                elif self.method == METHOD_ABS_DIFF:
                    image = self.absdiff((im1, im2))
                elif self.method == METHOD_FLOW_MASK:
                    image = self.flow_mask((im1, im2))
                elif self.method == METHOD_DIFF_MASK:
                    image = self.diff_mask((im1, im2))

                image = kimage.img_to_array(image)
                frames.append(image)

            images.append(frames)

        return np.array(images), batch[1]
