import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.transforms.functional as F1
from paddle.vision.transforms import transforms

from models.inception import inception_v3, BasicConv2d
from models.resnet import resnet50, resnet101
import logging

from utils import second_augment

EPSILON = 1e-12


class BAP(paddle.nn.Layer):
    def __init__(self, **kwargs):
        super(BAP, self).__init__()
        self.pool = None

    def forward(self, features, attentions):
        B, C, H, W = features.shape
        _, M, AH, AW = attentions.shape
        if AH != H or AW != W:
            F.interpolate(attentions, size=[H, W], mode="bilinear")

        # 此段程序用来替代einsum函数
        mat = []
        for i in range(B):  # batch 拆分
            cur_atm = attentions[i]  # 去除第一维
            cur_ftm = features[i]  # 去除第一维
            cur_atm = paddle.reshape(cur_atm, shape=[M, -1])  # 展开
            cur_ftm = paddle.reshape(cur_ftm, shape=[C, -1])  # 展开
            cur_ftm = paddle.transpose(cur_ftm, perm=[1, 0])  # 转置
            cur_feature_matrix = paddle.matmul(cur_atm, cur_ftm)  # 矩阵相乘，相当于点积
            mat.append(cur_feature_matrix)
        feature_matrix = paddle.stack(mat, axis=0) / float(H * W)

        # feature_matrix = paddle.einsum('imjk,injk->imn', attentions, features) / float(H * W)  # 动转静不能用
        feature_matrix = paddle.reshape(feature_matrix, [B, -1])

        # sign-sqrt
        feature_matrix = paddle.sign(feature_matrix) * paddle.sqrt(paddle.abs(feature_matrix) + 1e-12)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix, axis=-1)
        return feature_matrix


class CFGVC(paddle.nn.Layer):
    def __init__(self, num_classes, num_attentions=32, net_name='inception_mixed_6e', pretrained=False):
        super(CFGVC, self).__init__()
        self.num_classes = num_classes
        self.num_attentions = num_attentions
        self.net_name = net_name
        # Network Initialization
        if 'inception' in net_name:
            if net_name == 'inception_mixed_6e':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_6e()
                self.num_features = 768
            elif net_name == 'inception_mixed_7c':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_7c()
                self.num_features = 2048
            else:
                raise ValueError('Unsupported net: %s' % net_name)
        elif 'resnet' in net_name:
            self.features = resnet101(pretrained=pretrained).get_features()
            self.num_features = 512 * self.features[-1][-1].expansion
        else:
            raise ValueError('Unsupported net: %s' % net_name)

        # Attention Maps
        self.attentions = BasicConv2d(self.num_features, self.num_attentions, kernel_size=1)

        # Bilinear Attention Pooling
        self.bap = BAP(pool='GAP')

        # Classification Layer
        self.fc = nn.Linear(self.num_attentions * self.num_features, self.num_classes, bias_attr=False)

        logging.info(
            'CFGVC: using {} as feature extractor, num_classes: {}, num_attentions: {}'.format(net_name,
                                                                                               self.num_classes,
                                                                                               self.num_attentions))

    def forward(self, x, T):
        batch_size = x.shape[0]
        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.features(x)
        if T:
            feature_maps = second_augment(feature_maps)
        if self.net_name != 'inception_mixed_7c':
            attention_maps = self.attentions(feature_maps)
        else:
            attention_maps = feature_maps[:, :self.num_attentions, ...]

        feature_matrix = self.bap(feature_maps, attention_maps)

        # Classification
        p = self.fc(feature_matrix * 100.)

        # Generate Attention Map
        if self.training:
            # Randomly choose one of attention maps Ak
            attention_map = []
            for i in range(batch_size):
                attention_weights = paddle.sqrt(attention_maps[i].sum(axis=(1, 2)).detach() + EPSILON)
                attention_weights = F.normalize(attention_weights, p=1, axis=0)
                k_index = np.random.choice(self.num_attentions, 2, p=attention_weights.numpy())
                atm = paddle.stack([attention_maps[i, k_index[0], :, :], attention_maps[i, k_index[1], :, :]])
                attention_map.append(atm)  # 16,32,3,3
            attention_map = paddle.stack(attention_map)  # (B, 2, H, W) - one for cropping, the other for dropping
        else:
            # Object Localization Am = mean(Ak)
            attention_map = paddle.mean(attention_maps, axis=1, keepdim=True)  # (B, 1, H, W)

        return p, feature_matrix, attention_map, attention_maps

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}

        if len(pretrained_dict) == len(state_dict):
            logging.info('%s: All params loaded' % type(self).__name__)
        else:
            logging.info('%s: Some params were not loaded:' % type(self).__name__)
            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            logging.info(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))

        model_dict.update(pretrained_dict)
        super(CFGVC, self).set_state_dict(model_dict)
