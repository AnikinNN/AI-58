import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2


def plot_examples(images: torch.Tensor, landmarks: torch.Tensor):
    images = images.detach().cpu().numpy()
    landmarks = landmarks.detach().cpu().numpy()

    f = plt.figure(figsize=(6,6), dpi=300)
    for n,idx in enumerate(np.random.choice(np.arange(images.shape[0]), 9)):
        img = np.transpose(images[idx], (1,2,0)).astype(np.uint8)
        x,y,r = landmarks[idx,:]
        img2show = np.copy(img)
        img2show = cv2.circle(img2show, (x,y), r, (255,0,255), thickness=3)

        p = plt.subplot(3,3,n+1)
        plt.imshow(img2show)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()




def plot_examples_segmentation(images: torch.Tensor, segmaps: torch.Tensor, file_output: str = ""):
    images = images.detach().cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1)).astype(np.uint8)
    segmaps = segmaps.detach().cpu().numpy()
    segmaps = (np.transpose(segmaps, (0, 2, 3, 1))*255).astype(np.uint8)

    f = plt.figure(figsize=(8,4), dpi=300)
    for n,idx in enumerate(np.random.choice(np.arange(images.shape[0]), 4, replace=False)):
        img = images[idx]
        segmap = segmaps[idx, :, :, 0]

        p = plt.subplot(2,4,2*n+1)
        plt.imshow(img)
        plt.axis('off')
        p = plt.subplot(2, 4, 2*n + 2)
        plt.imshow(segmap, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
    plt.tight_layout()
    if file_output == "":
        plt.show()
    else:
        plt.savefig(file_output)
    plt.close()