import keras_video
import unittest
import os
import sys
import shutil
sys.path.insert(0, './src')


class TestDiscovery(unittest.TestCase):

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

    def test_find_classes(self):
        """ Check classe auto discovery """

        g = keras_video.VideoFrameGenerator(
            glob_pattern=os.path.join(self.testdir, '{classname}_*.ogv'))
        assert 'A' in g.classes
        assert 'B' in g.classes
        assert 'C' in g.classes

        assert g.files_count == 30

    def test_iterator(self):
        """ Check if generator is an iterator """
        g = keras_video.VideoFrameGenerator(
            batch_size=4,
            nb_frames=6,
            target_shape=(64, 64),
            glob_pattern=os.path.join(self.testdir, '{classname}_*.ogv'))

        # iterator object should be able to
        # use "next()" function
        x, y = next(g)

        assert x.shape == (4, 6, 64, 64, 3)
        assert y.shape == (4, 3)
