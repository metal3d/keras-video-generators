import os
import shutil
import sys
import unittest

import keras_video
from tensorflow import keras

sys.path.insert(0, "./src")


class TestOpticalFlow(unittest.TestCase):

    testdir = "test_vids"

    def setUp(self):
        dirname = self.testdir
        os.makedirs(dirname)

        def _write_zero(cl, i):
            shutil.copy(
                "tests/vidtest.ogv", os.path.join(self.testdir, "%s_%d.ogv" % (cl, i))
            )

        for i in range(10):
            for cl in ["A", "B", "C"]:
                _write_zero(cl, i)

    def tearDown(self):
        shutil.rmtree(self.testdir)

    def test_init(self):
        """ Check opticalflow init """
        gen = keras_video.OpticalFlowGenerator(
            glob_pattern=os.path.join(self.testdir, "{classname}_*.ogv")
        )
        assert len(gen.classes) == 3
        assert gen.files_count == 30

    def __get_with_method(self, method=keras_video.METHOD_ABS_DIFF):
        tr = keras.preprocessing.image.ImageDataGenerator(rotation_range=10)

        gen = keras_video.OpticalFlowGenerator(
            method=method,
            glob_pattern=os.path.join(self.testdir, "{classname}_*.ogv"),
            transformation=tr,
        )

        seq, labels = next(gen)
        assert seq.shape == (16, 5, 224, 224, 3)
        assert labels.shape == (16, 3)

    def test_absdiff(self):
        """ Check absdiff """
        self.__get_with_method(keras_video.METHOD_ABS_DIFF)

    def test_absdiffmask(self):
        """ Check absdiff masked """
        self.__get_with_method(keras_video.METHOD_DIFF_MASK)

    def test_opticalflow(self):
        """ Check opticalflow"""
        self.__get_with_method(keras_video.METHOD_OPTICAL_FLOW)

    def test_opticalflowmask(self):
        """ Check opticalflow masked """
        self.__get_with_method(keras_video.METHOD_FLOW_MASK)
