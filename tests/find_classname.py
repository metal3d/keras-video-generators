import sys
sys.path.insert(0, './src')

import unittest
import keras_video

class TestFindClassname(unittest.TestCase):

    def test_class_as_dirname(self):
        """ Check if classname in glob_pattern is detected when it's given as a directory name"""
        g = keras_video.VideoFrameGenerator(classes=['a'], glob_pattern='foo/bar/{classname}/*.avi')
        cl = g._get_classname('foo/bar/baz/zoo.avi')
        assert cl == 'baz'

    def test_class_in_filename(self):
        """ Check if classname in glob_pattern is detected when it's given in the filename """
        g = keras_video.VideoFrameGenerator(classes=['a'], glob_pattern='foo/bar/{classname}_*.avi')
        cl = g._get_classname('foo/bar/baz_0001.avi')
        assert cl == 'baz'

