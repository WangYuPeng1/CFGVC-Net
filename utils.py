from paddle.vision import ColorJitter
from paddle.vision.transforms import transforms
import paddle.vision.transforms.functional as F1
import paddle.nn.functional as F
import paddle.nn as nn
import paddle
import numpy as np
import random
from paddle.vision.transforms import BrightnessTransform
import paddle.fluid.layers as layers

# Center Loss for Attention Regularization
from ToPILImage import ToPILImage


class CenterLoss(paddle.nn.Layer):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.shape[0]


# Metric
class Metric(object):
    pass


class AverageMeter(Metric):
    def __init__(self, name='loss'):
        self.total_num = None
        self.scores = None
        self.name = name
        self.reset()

    def reset(self):
        self.scores = 0.
        self.total_num = 0.

    def __call__(self, batch_score, sample_num=1):
        self.scores += batch_score
        self.total_num += sample_num
        return self.scores / self.total_num


class TopKAccuracyMetric(Metric):
    def __init__(self, topk=(1,)):
        self.num_samples = None
        self.corrects = None
        self.name = 'topk_accuracy'
        self.topk = topk
        self.maxk = max(topk)
        self.reset()

    def reset(self):
        self.corrects = np.zeros(len(self.topk))
        self.num_samples = 0.

    def __call__(self, output, target):
        """Computes the precision@k for the specified values of k"""
        self.num_samples += target.shape[0]
        _, pred = output.topk(self.maxk, 1, True, True)
        pred = pred.t()
        target = paddle.reshape(target, [1, -1])
        correct = pred.equal(target.expand_as(pred))

        for i, k in enumerate(self.topk):
            temp = paddle.reshape(correct[:k], [-1])
            correct_k = temp.sum(0)
            self.corrects[i] += correct_k.item()

        return self.corrects * 100. / self.num_samples


# Callback
class Callback(object):
    def __init__(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, *args):
        pass


# augment function
def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.shape

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = nn.functional.interpolate(atten_map, size=(imgH, imgW), mode='BILINEAR') >= theta_c
            nonzero_indices = paddle.nonzero(crop_mask[0, 0, ...])
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            crop_images.append(
                nn.functional.interpolate(
                    images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                    size=(imgH, imgW), mode='BILINEAR'))
        crop_images = paddle.concat(crop_images, axis=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()

            drop_masks.append(nn.functional.interpolate(atten_map, size=(imgH, imgW), mode='BILINEAR') < theta_d)
        drop_masks = paddle.concat(drop_masks, axis=0)
        drop_images = images * drop_masks
        # plt.imshow(drop_images[0][0])
        # plt.imshow(np.array(drop_images[0]).transpose([1,2,0]))  #彩色
        # plt.show()
        return drop_images

    else:
        raise ValueError(
            'Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)


# transform in dataset
def getTransform(resize, mode='train'):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0] // 0.875), int(resize[1] // 0.875))),
            transforms.RandomCrop(resize),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=32. / 255, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0] // 0.875), int(resize[1] // 0.875))),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def second_augment(images):
    batches, _, imgH, imgW = images.shape
    second_images = []
    for batch_index in range(batches):
        image = images[batch_index]
        X2 = F1.hflip(image)
        second_images.append(X2)
    second_images = paddle.stack(second_images, axis=0)
    return second_images


def second_augment2(images):
    batches, _, imgH, imgW = images.shape
    second_images = []
    for batch_index in range(batches):
        image = images[batch_index]
        # X = transforms.RandomHorizontalFlip(1)(image)
        # X2 = BrightnessTransform(32. / 255)(X)
        X2 = transforms.Compose([
            # transforms.Resize(size=(int(448//0.875), int(448//0.875))),
            # transforms.RandomCrop((448, 448)),
            # transforms.CenterCrop((448, 448)),
            ToPILImage(),
            transforms.ColorJitter(brightness=32. / 255, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(image)
        second_images.append(X2)
    second_images = paddle.stack(second_images, axis=0)
    return second_images


# Calculate cross entropy loss, apply label smoothing if needed.
def cal_loss(pred, gold, smoothing=True):
    gold = paddle.reshape(gold, [-1])

    if smoothing:
        eps = 0.2
        n_class = pred.shape[1]
        one_hot_label = F.one_hot(gold, n_class)
        smooth_label = F.label_smooth(one_hot_label, epsilon=eps)
        log_prb = F.log_softmax(pred, axis=1)
        loss = -(smooth_label * log_prb).sum(axis=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss
