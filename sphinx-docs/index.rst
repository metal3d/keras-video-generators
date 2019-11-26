Welcome to Keras Video Generator's documentation!
=================================================

Installation
------------

Simply use ``pip`` to install ``keras_video`` package:

    pip install keras_video

Then you will be able to import the module:

.. code-block:: python

    import keras_video

    generator = keras_video.VideoFrameGenerator()


Note that classes are placed in several sub-modules in the package, but there are alias in the package file. That means that
instead of using ``keras_video.VideoFrameGenerator``, you can use ``keras_vide.VideoFrameGenerator``.

Package documentation
---------------------

.. automodule:: keras_video
    :members:

.. automodule:: keras_video.generator
    :members:

.. automodule:: keras_video.sliding
    :members:

.. automodule:: keras_video.flow
    :members:

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
