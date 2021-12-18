"""
VideoFrameGenerator - Simple Generator
--------------------------------------
A simple frame generator that takes distributed frames from
videos. It is useful for videos that are scaled from frame 0 to end
and that have no noise frames.
"""

import glob
import logging
import os
import re
from math import floor
from typing import Iterable, Optional

import cv2 as cv
import numpy as np
from tensorflow.keras.preprocessing.image import (ImageDataGenerator,
                                                  img_to_array)
from tensorflow.keras.utils import Sequence

log = logging.getLogger()


class VideoFrameGenerator(Sequence):  # pylint: disable=too-many-instance-attributes
    """
    Create a generator that return batches of frames from video
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
    - glob_pattern: string, directory path with '{classname}' inside that \
        will be replaced by one of the class list
    - use_header: bool, default to True to use video header to read the \
        frame count if possible

    - seed: int, default to None, keep the seed value for split

    You may use the "classes" property to retrieve the class list afterward.
    The generator has that properties initialized:
    - classes_count: number of classes that the generator manages
    - files_count: number of video that the generator can provides
    - classes: the given class list
    - files: the full file list that the generator will use, this \
        is usefull if you want to remove some files that should not be \
        used by the generator.
    """

    def __init__(  # pylint: disable=too-many-statements,too-many-locals,too-many-branches,too-many-arguments
        self,
        rescale: float = 1 / 255.0,
        nb_frames: int = 5,
        classes: list = None,
        batch_size: int = 16,
        use_frame_cache: bool = False,
        target_shape: tuple = (224, 224),
        shuffle: bool = True,
        transformation: Optional[ImageDataGenerator] = None,
        split_test: float = None,
        split_val: float = None,
        nb_channel: int = 3,
        glob_pattern: str = "./videos/{classname}/*.avi",
        use_headers: bool = True,
        seed=None,
        **kwargs,
    ):

        self.glob_pattern = glob_pattern

        # should be only RGB or Grayscale
        assert nb_channel in (1, 3)

        if classes is None:
            classes = self._discover_classes()

        # we should have classes
        if len(classes) == 0:
            log.warn(
                "You didn't provide classes list or "
                "we were not able to discover them from "
                "your pattern.\n"
                "Please check if the path is OK, and if the glob "
                "pattern is correct.\n"
                "See https://docs.python.org/3/library/glob.html"
            )

        # shape size should be 2
        assert len(target_shape) == 2

        # split factor should be a propoer value
        if split_val is not None:
            assert 0.0 < split_val < 1.0

        if split_test is not None:
            assert 0.0 < split_test < 1.0

        self.use_video_header = use_headers

        # then we don't need None anymore
        split_val = split_val if split_val is not None else 0.0
        split_test = split_test if split_test is not None else 0.0

        # be sure that classes are well ordered
        classes.sort()

        self.rescale = rescale
        self.classes = classes
        self.batch_size = batch_size
        self.nbframe = nb_frames
        self.shuffle = shuffle
        self.target_shape = target_shape
        self.nb_channel = nb_channel
        self.transformation = transformation
        self.use_frame_cache = use_frame_cache

        self._random_trans = []
        self.__frame_cache = {}
        self.files = []
        self.validation = []
        self.test = []

        _validation_data = kwargs.get("_validation_data", None)
        _test_data = kwargs.get("_test_data", None)
        np.random.seed(seed)

        if _validation_data is not None:
            # we only need to set files here
            self.files = _validation_data

        elif _test_data is not None:
            # we only need to set files here
            self.files = _test_data
        else:
            self.__split_from_vals(
                split_val, split_test, classes, shuffle, glob_pattern
            )

        # build indexes
        self.files_count = len(self.files)
        self.indexes = np.arange(self.files_count)
        self.classes_count = len(classes)

        # to initialize transformations and shuffle indices
        if "no_epoch_at_init" not in kwargs:
            self.on_epoch_end()

        kind = "train"
        if _validation_data is not None:
            kind = "validation"
        elif _test_data is not None:
            kind = "test"

        self._current = 0
        self._framecounters = {}
        print(
            "Total data: %d classes for %d files for %s"
            % (self.classes_count, self.files_count, kind)
        )

    def count_frames(self, cap, name, force_no_headers=False):
        """Count number of frame for video
        if it's not possible with headers"""
        if not force_no_headers and name in self._framecounters:
            return self._framecounters[name]

        total = cap.get(cv.CAP_PROP_FRAME_COUNT)

        if force_no_headers or total < 0:
            # headers not ok
            total = 0
            # TODO: we're unable to use CAP_PROP_POS_FRAME here
            # so we open a new capture to not change the
            # pointer position of "cap"
            capture = cv.VideoCapture(name)
            while True:
                grabbed, _ = capture.read()
                if not grabbed:
                    # rewind and stop
                    break
                total += 1

        # keep the result
        self._framecounters[name] = total

        return total

    def __split_from_vals(self, split_val, split_test, classes, shuffle, glob_pattern):
        """ Split validation and test set """

        if split_val == 0 or split_test == 0:
            # no splitting, do the simplest thing
            for cls in classes:
                self.files += glob.glob(glob_pattern.format(classname=cls))
            return

        # else, there is some split to do
        for cls in classes:
            files = glob.glob(glob_pattern.format(classname=cls))
            nbval = 0
            nbtest = 0
            info = []

            # generate validation and test indexes
            indexes = np.arange(len(files))

            if shuffle:
                np.random.shuffle(indexes)

            nbtrain = 0
            if 0.0 < split_val < 1.0:
                nbval = int(split_val * len(files))
                nbtrain = len(files) - nbval

                # get some sample for validation_data
                val = np.random.permutation(indexes)[:nbval]

                # remove validation from train
                indexes = np.array([i for i in indexes if i not in val])
                self.validation += [files[i] for i in val]
                info.append("validation count: %d" % nbval)

            if 0.0 < split_test < 1.0:
                nbtest = int(split_test * nbtrain)
                nbtrain = len(files) - nbval - nbtest

                # get some sample for test_data
                val_test = np.random.permutation(indexes)[:nbtest]

                # remove test from train
                indexes = np.array([i for i in indexes if i not in val_test])
                self.test += [files[i] for i in val_test]
                info.append("test count: %d" % nbtest)

            # and now, make the file list
            self.files += [files[i] for i in indexes]
            print("class %s, %s, train count: %d" % (cls, ", ".join(info), nbtrain))

    def _discover_classes(self):
        pattern = os.path.realpath(self.glob_pattern)
        pattern = re.escape(pattern)
        pattern = pattern.replace("\\{classname\\}", "(.*?)")
        pattern = pattern.replace("\\*", ".*")

        files = glob.glob(self.glob_pattern.replace("{classname}", "*"))
        classes = set()
        for filename in files:
            filename = os.path.realpath(filename)
            classname = re.findall(pattern, filename)[0]
            classes.add(classname)

        return list(classes)

    def next(self):
        """ Return next element"""
        elem = self[self._current]
        self._current += 1
        if self._current == len(self):
            self._current = 0
            self.on_epoch_end()

        return elem

    def get_validation_generator(self):
        """ Return the validation generator if you've provided split factor """
        return self.__class__(
            nb_frames=self.nbframe,
            nb_channel=self.nb_channel,
            target_shape=self.target_shape,
            classes=self.classes,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            rescale=self.rescale,
            glob_pattern=self.glob_pattern,
            use_headers=self.use_video_header,
            _validation_data=self.validation,
        )

    def get_test_generator(self):
        """ Return the validation generator if you've provided split factor """
        return self.__class__(
            nb_frames=self.nbframe,
            nb_channel=self.nb_channel,
            target_shape=self.target_shape,
            classes=self.classes,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            rescale=self.rescale,
            glob_pattern=self.glob_pattern,
            use_headers=self.use_video_header,
            _test_data=self.test,
        )

    def on_epoch_end(self):
        """ Called by Keras after each epoch """

        if self.transformation is not None:
            self._random_trans = []
            for _ in range(self.files_count):
                self._random_trans.append(
                    self.transformation.get_random_transform(self.target_shape)
                )

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return int(np.floor(self.files_count / self.batch_size))

    def __getitem__(self, index):
        classes = self.classes
        shape = self.target_shape
        nbframe = self.nbframe

        labels = []
        images = []

        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        transformation = None

        for i in indexes:

            video = self.files[i]
            classname = self._get_classname(video)

            # create a label array and set 1 to the right column
            label = np.zeros(len(classes))
            col = classes.index(classname)
            label[col] = 1.0

            if video not in self.__frame_cache:
                frames = self._get_frames(
                    video, nbframe, shape, force_no_headers=not self.use_video_header
                )
                if frames is None:
                    # avoid failure, nevermind that video...
                    continue

                # add to cache
                if self.use_frame_cache:
                    self.__frame_cache[video] = frames

            else:
                frames = self.__frame_cache[video]

            # apply transformation
            # if provided
            if self.transformation is not None:
                transformation = self._random_trans[i]
                frames = [
                    self.transformation.apply_transform(frame, transformation)
                    if transformation is not None
                    else frame
                    for frame in frames
                ]

            # add the sequence in batch
            images.append(frames)
            labels.append(label)

        return np.array(images), np.array(labels)

    def _get_classname(self, video: str) -> str:
        """ Find classname from video filename following the pattern """

        # work with real path
        video = os.path.realpath(video)
        pattern = os.path.realpath(self.glob_pattern)

        # remove special regexp chars
        pattern = re.escape(pattern)

        # get back "*" to make it ".*" in regexp
        pattern = pattern.replace("\\*", ".*")

        # use {classname} as a capture
        pattern = pattern.replace("\\{classname\\}", "(.*?)")

        # and find all occurence
        classname = re.findall(pattern, video)[0]
        return classname

    def _get_frames(
        self, video, nbframe, shape, seq_time=0, force_no_headers=False
    ) -> Optional[Iterable]:
        cap = cv.VideoCapture(video)
        total_frames = self.count_frames(cap, video, force_no_headers)
        fps = cap.get(cv.CAP_PROP_FPS)
        orig_total = total_frames

        if total_frames % 2 != 0:
            total_frames += 1

        frame_step = floor(total_frames / (nbframe - 1)) if seq_time ==0 else floor((seq_time * fps) / (nbframe - 1))
        # TODO: fix that, a tiny video can have a frame_step that is
        # under 1
        frame_step = max(1, frame_step)
        frames = []
        frame_i = 0

        while True:
            grabbed, frame = cap.read()
            if not grabbed:
                break

            self.__add_and_convert_frame(
                frame, frame_i, frames, orig_total, shape, frame_step
            )

            if len(frames) == nbframe:
                break

        cap.release()

        if not force_no_headers and len(frames) != nbframe:
            # There is a problem here
            # That means that frame count in header is wrong or broken,
            # so we need to force the full read of video to get the right
            # frame counter
            return self._get_frames(video, nbframe, shape, force_no_headers=True)

        if force_no_headers and len(frames) != nbframe:
            # and if we really couldn't find the real frame counter
            # so we return None. Sorry, nothing can be done...
            log.error(
                f"Frame count is not OK for video {video}, "
                f"{total_frames} total, {len(frames)} extracted"
            )
            return None

        return np.array(frames)

    def __add_and_convert_frame(  # pylint: disable=too-many-arguments
        self, frame, frame_i, frames, orig_total, shape, frame_step
    ):
        frame_i += 1
        if frame_i in (1, orig_total) or frame_i % frame_step == 0:
            # resize
            frame = cv.resize(frame, shape)

            # use RGB or Grayscale ?
            frame = (
                cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                if self.nb_channel == 3
                else cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            )

            # to np
            frame = img_to_array(frame) * self.rescale

            # keep frame
            frames.append(frame)
