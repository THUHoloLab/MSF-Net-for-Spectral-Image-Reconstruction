import torch.utils.data as data
import numpy as np
import hdf5storage as hdf5
import random
import torch
import glob


def load_img(filepath):
    x = hdf5.loadmat(filepath)
    x = x['HSI']
    x = torch.tensor(x).float()
    return x


# train dataset
class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, image_size):
        super(DatasetFromFolder, self).__init__()
        self.image_size = image_size
        self.image_filenames = glob.glob(image_dir + '/*/*.mat')
        self.xs = []
        for img in self.image_filenames:
            self.xs.append(load_img(img))

    def __getitem__(self, ind):
        X = self.xs[ind]
        w = np.random.randint(0, X.shape[0] - self.image_size+1)
        h = np.random.randint(0, X.shape[1] - self.image_size+1)
        X = X[w:w + self.image_size, h:h + self.image_size, :]
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        X = torch.rot90(X, rotTimes, [0, 1])
        # Random vertical Flip
        for j in range(vFlip):
            X = X.flip(1)
        # Random Horizontal Flip
        for j in range(hFlip):
            X = X.flip(0)
        X = X.permute(2, 0, 1)
        return X

    def __len__(self):
        return len(self.image_filenames)

def load_test_dataset(image_dir):
    image_filenames = glob.glob(image_dir + '/*/*.mat')
    xs = []
    for img in image_filenames:
        xs.append(load_img(img))
    test_dataset=torch.stack(xs).float()
    test_dataset=test_dataset.permute(0,3,1,2)
    return test_dataset