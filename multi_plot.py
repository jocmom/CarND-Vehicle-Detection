import math
import itertools
import matplotlib.pyplot as plt

# plotting helper functions
def plot_images(images, label=None, n_cols=5, cmap=None, savefig=None):
    """
    plot images in n_cols columns
    :param images
    :param label image labels
    :param n_cols number of columns
    """
    n_rows = math.ceil(len(images)/n_cols)

    plt.figure(figsize=(20, n_rows*10/n_cols))
    for i, image in enumerate(images):
        #image = image.squeeze()
        plt.subplot(n_rows, n_cols, i+1)
        if label != None:
            plt.title("Output " + str(label[i]))
        plt.imshow(image, aspect='equal', cmap=cmap)
    if savefig != None:
        plt.savefig(savefig)
    plt.show()

def plot_compare_images(left_images, right_images, cmap=None, savefig=None):
    """
    compare two image list side by side
    """
    assert len(left_images) == len(right_images)
    mixed_images = list(itertools.chain.from_iterable(zip(left_images, right_images)))
    plot_images(mixed_images, n_cols=2, cmap=cmap, savefig=savefig)

def plot_histogram(labels, n_labels):
    """
    Exploration of the label distribution
    """
    plt.hist(labels, n_labels)
    plt.xlabel('Labels')
    plt.ylabel('Label Count')
    plt.title('Histogram')
    plt.show()
