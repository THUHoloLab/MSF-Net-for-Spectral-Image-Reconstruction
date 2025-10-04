import numpy as np

def get_coded_aperture(image_size, channel):
    np.random.seed(10)
    mask = np.random.randint(0, 2, [image_size, image_size + channel - 1])
    mask_3D = np.zeros([channel, image_size, image_size])
    for i in range(0, channel):
        mask_3D[i, :, :] = mask[:, i:i + image_size]
    return mask_3D
