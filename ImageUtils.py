import numpy as np

"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    depth_major = record.reshape((3, 32, 32))
    image = np.transpose(depth_major, [1, 2, 0])

    ### END CODE HERE

    image = preprocess_image(image, training) # If any.
    image = np.transpose(image, [2, 0, 1])

    return image


def preprocess_image(image, training):
    """Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3]. The processed image.
    """
    ### YOUR CODE HERE
    if training:
        image = np.pad(image, pad_width=((4, 4), (4, 4), (0, 0)))
        rx = np.random.randint(8)
        ry = np.random.randint(8)
        crp_img = image[rx:rx + 32, ry:ry + 32, :]
        rf = np.random.randint(2)
        if (rf == 0):
            crp_img = np.fliplr(crp_img)
        image = crp_img
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    std = np.std(image, axis=(0, 1), keepdims=True)
    image = (image - mean) / std
    ### END CODE HERE

    return image


# Other functions
### YOUR CODE HERE

### END CODE HERE