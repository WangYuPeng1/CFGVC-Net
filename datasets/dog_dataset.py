import os
import numpy as np
from PIL import Image
from scipy.io import loadmat
from paddle.io import Dataset
from utils import getTransform
import dataset_path_config

DATAPATH = dataset_path_config.dog_dataset_path


class DogDataset(Dataset):

    def __init__(self, mode='train', resize=(448, 448)):
        super(DogDataset, self).__init__()
        self.mode = mode
        assert self.mode in ['train', 'val', 'test'], "mode should be 'train' or 'test', but got {}".format(self.mode)
        self.resize = resize
        self.num_classes = 120

        if mode == 'train':
            list_path = os.path.join(DATAPATH, 'train_list.mat')
        else:
            list_path = os.path.join(DATAPATH, 'test_list.mat')

        list_mat = loadmat(list_path)
        self.images = [f.item().item() for f in list_mat['file_list']]
        self.labels = [f.item() for f in list_mat['labels']]

        self.transform = getTransform(self.resize, self.mode)

    def __getitem__(self, item):
        image = Image.open(os.path.join(DATAPATH, 'Images', self.images[item])).convert('RGB')  # (C, H, W)
        image = self.transform(image)
        label = np.int64(self.labels[item]-1)

        return image, label  # count begin from zero

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    ds = DogDataset('train')
    print(len(ds))
    for i in range(0, 100):
        image, label = ds[i]
        print(image.shape, label)
