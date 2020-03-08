import keras_video
import unittest
import os
import sys
import shutil
sys.path.insert(0, './src')


class TestSplit(unittest.TestCase):

    testdir = 'test_vids'

    def setUp(self):
        dirname = self.testdir
        os.makedirs(dirname)

        def _write_zero(cl, i):
            shutil.copy(
                'tests/vidtest.ogv',
                os.path.join(self.testdir, '%s_%d.ogv' % (cl, i))
            )

        for i in range(10):
            for cl in ['A', 'B', 'C']:
                _write_zero(cl, i)

    def tearDown(self):
        shutil.rmtree(self.testdir)

    def __split(self,
                kind=keras_video.VideoFrameGenerator,
                vc=0,
                tc=0):
        pattern = os.path.join(self.testdir, '{classname}_*.ogv')
        gen = kind(
            glob_pattern=pattern,
            split_test=.2,
            split_val=.3)
        valid = gen.get_validation_generator()
        test = gen.get_test_generator()

        assert valid.files_count == vc
        assert test.files_count == tc

    def test_videoframegenerator_split(self):
        """ Check spliting VideoFrameGenerator """
        self.__split(keras_video.VideoFrameGenerator, 9, 3)

    def test_slidinggenerator_split(self):
        """ Check splitint SlidingFrameGenerator """
        self.__split(keras_video.SlidingFrameGenerator, 9, 3)

    def test_flowgenerator_split(self):
        """ Check splitint OpticalFlowGenerator """
        self.__split(keras_video.OpticalFlowGenerator, 9, 3)
