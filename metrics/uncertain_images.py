import numpy as np
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import pyplot as plt

class Metric():
    def __init__(self, **kwargs):
        self.requires = ['features', 'target_labels', 'dataloader']
        self.name     = 'uncertain_images'

    def __call__(self, features, target_labels, dataloader):
        # Setup image grid
        fig = plt.figure(figsize=(20., 20.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(10, 10),  # creates 2x2 grid of axes
                         axes_pad=0.1,  # pad between axes in inch.
                         )

        # Find most certain/uncertain images per class
        imgs = []
        chosenclasses = np.random.choice(np.unique(target_labels), size=10, replace=False)
        for lab in chosenclasses:
            ids = np.array([i for i in np.arange(len(target_labels)) if target_labels[i] == lab])
            norms = torch.norm(torch.from_numpy(features[ids]), dim=1)
            order = torch.argsort(norms)
            first5 = ids[order[:5]]
            last5 = ids[order[-5:]]
            all = np.concatenate((first5, last5))

            # get images
            for i, id in enumerate(all):
                _, img, _ = dataloader.dataset.__getitem__(id)
                imgs.append(img)

        # Plot images
        for ax, im in zip(grid, imgs):
            # Iterating over the grid returns the Axes.
            ax.imshow(torch.minimum(torch.ones(1), torch.maximum(torch.zeros(1), im.permute(1, 2, 0) / 3 + 0.5)))
        fig.savefig("uncertain_images.png")

        return 0
