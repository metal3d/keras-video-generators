"""
Sliding frames
--------------

That module provides the SlidingFrameGenerator that is helpful
to get more sequence from one video file. The goal is to provide decayed
sequences for the same action.


"""

import os
import numpy as np
import cv2 as cv
from math import floor
from .generator import VideoFrameGenerator


class SlidingFrameGenerator(VideoFrameGenerator):
    """
    SlidingFrameGenerator is useful to get several sequence of
    the same "action" by sliding the cursor of video. For example, with a
    video that have 60 frames using 30 frames per second, and if you want
    to pick 6 frames, the generator will return:

    - one sequence with frame ``[ 0,  5, 10, 15, 20, 25]``
    - then ``[ 1,  6, 11, 16, 21, 26])``
    - and so on to frame 30

    If you set `sequence_time` parameter, so the sequence will be reduce to
    the given time.

    params:

    - sequence_time: int seconds of the sequence to fetch, if None, the entire \
        vidoe time is used

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

    def __init__(self, *args, sequence_time: int = None, **kwargs):
        super().__init__(no_epoch_at_init=True, *args, **kwargs)
        self.sequence_time = sequence_time

        self.sample_count = 0
        self.vid_info = []
        self.__frame_cache = {}
        self.__init_length()
        self.on_epoch_end()

    def __init_length(self):
        count = 0
        print("Checking files to find possible sequences, please wait...")
        for filename in self.files:
            cap = cv.VideoCapture(filename)
            fps = cap.get(cv.CAP_PROP_FPS)
            frame_count = self.count_frames(cap, filename)
            cap.release()

            if self.sequence_time is not None:
                seqtime = int(fps*self.sequence_time)
            else:
                seqtime = int(frame_count)

            stop_at = int(seqtime - self.nbframe)
            step = np.ceil(seqtime / self.nbframe).astype(np.int) - 1
            i = 0
            while i <= frame_count - stop_at:
                self.vid_info.append({
                    'id': count,
                    'name': filename,
                    'frame_count': int(frame_count),
                    'frames': np.arange(i, i + stop_at)[::step][:self.nbframe],
                    'fps': fps,
                })
                count += 1
                i += 1

        print("For %d files, I found %d possible sequence samples" %
              (self.files_count, len(self.vid_info)))
        self.indexes = np.arange(len(self.vid_info))

    def on_epoch_end(self):
        # prepare transformation to avoid __getitem__ to reinitialize them
        if self.transformation is not None:
            self._random_trans = []
            for _ in range(len(self.vid_info)):
                self._random_trans.append(
                    self.transformation.get_random_transform(self.target_shape)
                )

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.vid_info) / self.batch_size))

    def get_validation_generator(self):
        """ Return the validation generator if you've provided split factor """
        return self.__class__(
            sequence_time=self.sequence_time,
            nb_frames=self.nbframe,
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
            sequence_time=self.sequence_time,
            nb_frames=self.nbframe,
            nb_channel=self.nb_channel,
            target_shape=self.target_shape,
            classes=self.classes,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            rescale=self.rescale,
            glob_pattern=self.glob_pattern,
            _test_data=self.test)

    def __getitem__(self, idx):
        classes = self.classes
        shape = self.target_shape
        nbframe = self.nbframe

        labels = []
        images = []

        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]

        transformation = None

        for i in indexes:
            # prepare a transformation if provided
            if self.transformation is not None:
                transformation = self._random_trans[i]

            vid = self.vid_info[i]
            video = vid.get('name')
            classname = self._get_classname(video)

            # create a label array and set 1 to the right column
            label = np.zeros(len(classes))
            col = classes.index(classname)
            label[col] = 1.

            if vid['id'] not in self.__frame_cache:
                frames = self._get_frames(video, nbframe, shape)
            else:
                frames = self.__frame_cache[vid['id']]

            # apply transformation
            if transformation is not None:
                frames = [self.transformation.apply_transform(
                    frame, transformation) for frame in frames]

            # add the sequence in batch
            images.append(frames)
            labels.append(label)

        return np.array(images), np.array(labels)
