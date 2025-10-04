import numpy as np
import hdf5storage as hdf5

def get_coded_aperture(image_size, channel):
    np.random.seed(10)
    mask = np.random.randint(0, 2, [image_size, image_size + channel - 1])
    mask_3D = np.zeros([channel, image_size, image_size])
    for i in range(0, channel):
        mask_3D[i, :, :] = mask[:, i:i + image_size]
    return mask_3D

if __name__ == '__main__':
    mask_CASSI=get_coded_aperture(256,29)
    print(mask_CASSI.shape)
    mask_gray = np.ones_like(mask_CASSI)
    print(mask_gray.shape)

    mask_CASSI = {"mask_CASSI": mask_CASSI}
    hdf5.savemat('mask_CASSI.mat', mask_CASSI)
    mask_gray = {"mask_gray": mask_gray}
    hdf5.savemat('mask_gray.mat', mask_gray)