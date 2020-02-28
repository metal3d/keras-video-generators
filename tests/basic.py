
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
            with open(os.path.join(dirname, '%s_%d.avi' % (cl, i)), 'w') as f:
                f.write("0")

        for i in range(10):
            for cl in ['A', 'B', 'C']:
                _write_zero(cl, i)

    def tearDown(self):
        shutil.rmtree(self.testdir)

    def test_find_classes(self):
        """ Check classe auto discovery """

        g = keras_video.VideoFrameGenerator(
            glob_pattern=os.path.join(self.testdir, '{classname}_*.avi'))
        assert 'A' in g.classes
        assert 'B' in g.classes
        assert 'C' in g.classes

        assert g.files_count == 30
