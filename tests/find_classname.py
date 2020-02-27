import keras_video
import unittest
import os
import sys
sys.path.insert(0, './src')


class TestFindClassname(unittest.TestCase):

    def test_class_as_dirname(self):
        """ Check if classname in directory """
        g = keras_video.VideoFrameGenerator(
            classes=['a'], glob_pattern='foo/bar/{classname}/*.avi')
        cl = g._get_classname('foo/bar/baz/zoo.avi')
        assert cl == 'baz'

    def test_class_in_filename(self):
        """ Check if classname in filename """
        g = keras_video.VideoFrameGenerator(
            classes=['a'], glob_pattern='foo/bar/{classname}_*.avi')
        cl = g._get_classname('foo/bar/baz_0001.avi')
        assert cl == 'baz'

    def test_class_for_windows(self):
        """ Work with windows path """
        current = os.name
        os.name = 'windows'

        g = keras_video.VideoFrameGenerator(
            classes=['a'], glob_pattern='foo\\bar\\{classname}_*.avi')
        cl = g._get_classname('foo\\bar\\baz_0001.avi')

        os.name = current
        assert cl == 'baz'

    def test_class_with_point(self):
        """ Check with ./ in path """
        g = keras_video.VideoFrameGenerator(
            classes=['a'], glob_pattern='./foo/bar/{classname}_*.avi')
        cl = g._get_classname('foo/bar/baz_0001.avi')
        assert cl == 'baz'
