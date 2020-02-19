"""
Utils module to provide some nice functions to develop with Keras and
Video sequences.
"""

import numpy as np
import random as rnd
import matplotlib.pyplot as plt


def show_sample(g, index=0, random=False, row_width=22, row_height=5):
    """ Displays a batch using matplotlib.

    params:

    - g: keras video generator
    - index: integer index of batch to see (overriden if random is True)
    - random: boolean, if True, take a random batch from the generator
    - row_width: integer to give the figure height
    - row_height: integer that represents one line of image, it is multiplied by \
    the number of sample in batch.
    """
    total = len(g)
    if random:
        sample = rnd.randint(0, total)
    else:
        sample = index

    assert index < len(g)
    sample = g[sample]
    sequences = sample[0]
    labels = sample[1]

    rows = len(sequences)
    index = 1
    plt.figure(figsize=(row_width, row_height*rows))
    for batchid, sequence in enumerate(sequences):
        classid = np.argmax(labels[batchid])
        classname = g.classes[classid]
        cols = len(sequence)
        for image in sequence:
            plt.subplot(rows, cols, index)
            plt.title(classname)
            plt.imshow(image)
            plt.axis('off')
            index += 1
    plt.show()
