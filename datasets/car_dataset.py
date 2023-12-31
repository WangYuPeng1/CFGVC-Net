import os
import numpy as np
from PIL import Image
from scipy.io import loadmat
from paddle.io import Dataset
from utils import getTransform
import dataset_path_config

DATAPATH = dataset_path_config.car_dataset_path


class CarDataset(Dataset):
    def __init__(self, mode='train', resize=(448, 448)):
        super(CarDataset, self).__init__()
        self.mode = mode
        assert self.mode in ['train', 'val', 'test'], "mode should be 'train' or 'test', but got {}".format(self.mode)
        self.resize = resize
        self.num_classes = 196

        if mode == 'train':
            list_path = os.path.join(DATAPATH, 'devkit', 'cars_train_annos.mat')
            self.image_path = os.path.join(DATAPATH, 'cars_train')
        else:
            list_path = os.path.join(DATAPATH, 'cars_test_annos_withlabels.mat')
            self.image_path = os.path.join(DATAPATH, 'cars_test')

        list_mat = loadmat(list_path)
        self.images = [f.item() for f in list_mat['annotations']['fname'][0]]
        self.labels = [f.item() for f in list_mat['annotations']['class'][0]]

        self.transform = getTransform(self.resize, self.mode)

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.image_path, self.images[item])).convert('RGB')  # (C, H, W)
        image = self.transform(image)
        label = np.int64(self.labels[item] - 1)

        return image, label

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ds = CarDataset('train')
    print(len(ds))
    for i in range(0, 10):
        image, label = ds[i]
        plt.imshow(image[0])
        plt.show()
