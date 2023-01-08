from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt


def plot_images(images):

    # Create a figure with a grid of subplots
    fig, axs = plt.subplots(nrows=1, ncols=10, figsize=(20, 2), facecolor='white')

    # Loop through the subplots and plot the images
    for ax, im in zip(axs.flat, images):
        ax.imshow(im, cmap='gray')
        ax.axis('off')

    # Display the figure
    plt.show()


def augment(image, rotation_angle=10, shift_scale=0.1):

    # Get the height and width of the image
    h, w = image.shape

    # Choose a random angle of rotation
    angle = np.random.uniform(-rotation_angle, rotation_angle)

    # Rotate the image by the chosen angle, using nearest-neighbor interpolation
    image = ndimage.rotate(image, angle, reshape=False)

    # Choose random amounts of horizontal and vertical shifting
    shift_x = np.random.uniform(-shift_scale, shift_scale) * w
    shift_y = np.random.uniform(-shift_scale, shift_scale) * h

    # Shift the image by the chosen amounts, using nearest-neighbor interpolation
    image = ndimage.shift(image, (shift_y, shift_x), order=0)

    # Return the augmented image
    return image
