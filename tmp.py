import cv2 as cv
import numpy as np

def show(img):
    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()



def add_cloud(rgb):
    cloud = cv.imread('cloud/cloud_000000.png', -1)

    alpha = cloud[:, :, 3] / 255.
    alpha = np.broadcast_to(alpha[:, :, None], alpha.shape + (3,))
    cloud_rgb = (1. - alpha) * rgb + alpha * cloud[:, :, :3]
    cloud_rgb = np.clip(cloud_rgb, 0., 255.)

    cv.imwrite('img_add_cloud.png', cloud_rgb)

img = cv.imread('data/RGB/img_000001.png', -1)
add_cloud(img)