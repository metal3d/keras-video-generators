import os
import shutil
import sys
import unittest

import keras_video
from tensorflow import keras

sys.path.insert(0, "./src")


class TestSlinding(unittest.TestCase):

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
        """ Check if slinding generator init """
        g = keras_video.SlidingFrameGenerator(
            glob_pattern=os.path.join(self.testdir, "{classname}_*.ogv")
        )
        assert "A" in g.classes
        assert "B" in g.classes
        assert "C" in g.classes

        assert g.files_count == 30

        # check get item
        seq, labels = next(g)
        assert seq.shape == (16, 5, 224, 224, 3)
        assert labels.shape == (16, 3)

    def test_with_transformation(self):
        """ Check if transformation works with slinding frame generator """
        tr = keras.preprocessing.image.ImageDataGenerator(rotation_range=10)
        g = keras_video.SlidingFrameGenerator(
            transformation=tr,
            glob_pattern=os.path.join(self.testdir, "{classname}_*.ogv"),
        )

        # check get item
        seq, labels = next(g)
        assert seq.shape == (16, 5, 224, 224, 3)
        assert labels.shape == (16, 3)
