import os
import logging
import argparse
import paddle
from datasets import getDataset
from paddle.io import DataLoader
from models.cfgvc import CFGVC
from utils import AverageMeter, TopKAccuracyMetric, batch_augment, cal_loss
import paddle.nn.functional as F


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="car", type=str, help="Which dataset you want to verify? bird, car, "
                                                                   "aircraft, dog, bird_tiny")
    parser.add_argument("--test-log-path", default="FGVC/", type=str)
    parser.add_argument("--batch-size", default=10, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--input-size", default=(448, 448), type=tuple)
    parser.add_argument("--net-name", default='resnet50', type=str, help="feature extractor")
    parser.add_argument("--num-attentions", default=32, type=int, help="number of attention maps")
    args = parser.parse_args()

    return args


def val():
    # read the parameters
    global net_state_dict, epoch_acc
    args = getArgs()

    # logging config
    logging.basicConfig(filename=os.path.join(args.test_log_path + args.dataset, 'test.log'), filemode='w',
                        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s', level=logging.INFO)
    logging.info('Current Testing Model: {}'.format(args.dataset))

    # read the dataset
    train_dataset, val_dataset = getDataset(args.dataset, args.input_size)
    train_loader, val_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers), \
                               DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # output the dataset info
    logging.info('Dataset Name:{dataset_name}, Val:[{val_num}]'.format(dataset_name=args.dataset, train_num=len(train_dataset), val_num=len(val_dataset)))
    logging.info('Batch Size:[{0}], Train Batches:[{1}], Val Batches:[{2}]'.format(args.batch_size, len(train_loader), len(val_loader)))

    # loss and metric
    loss_container = AverageMeter(name='loss')
    raw_metric = TopKAccuracyMetric(topk=(1, 5))
    ref_metric = TopKAccuracyMetric(topk=(1, 5))
    num_classes = train_dataset.num_classes

    # load the network and parameters
    net = CFGVC(num_classes=num_classes, num_attentions=args.num_attentions, net_name=args.net_name, pretrained=False)
    if args.dataset == 'bird':
        net_state_dict = paddle.load("FGVC/bird/bird_model.pdparams")
    if args.dataset == 'aircraft':
        net_state_dict = paddle.load("FGVC/aircraft/aircraft_model.pdparams")
    if args.dataset == 'car':
        net_state_dict = paddle.load("FGVC/car/car_model.pdparams")
    if args.dataset == 'dog':
        net_state_dict = paddle.load("FGVC/dog/dog_model.pdparams")
    if args.dataset == 'nabirds':
        net_state_dict = paddle.load("FGVC/bird_tiny/nabirds_model.pdparams")
    net.set_dict(net_state_dict)
    net.eval()

    # loss function
    # cross_entropy_loss = paddle.nn.CrossEntropyLoss()
    cross_entropy_loss = cal_loss

    # start to val
    logs = {}
    for i, (X, y) in enumerate(val_loader):
        # Raw Image
        y_pred_raw, _, attention_map, _ = net(X, False)

        # Object Localization and Refinement
        crop_images = batch_augment(X, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
        y_pred_crop, _, _, _ = net(crop_images, False)

        # Final prediction
        y_pred = (y_pred_raw + y_pred_crop) / 2.
        # loss
        batch_loss = cross_entropy_loss(y_pred, y)
        epoch_loss = loss_container(batch_loss.item())

        # metrics: top-1,5 error
        epoch_raw = raw_metric(y_pred_crop, y)
        epoch_acc = ref_metric(y_pred, y)

    logs['val_{}'.format(loss_container.name)] = epoch_loss
    logs['val_{}'.format(raw_metric.name)] = epoch_raw
    logs['val_{}'.format(ref_metric.name)] = epoch_acc
    batch_info = 'Val Loss {:.4f}, Val Acc ({:.2f}, {:.2f})，Ref Acc ({:.2f}, {:.2f})'.format(epoch_loss, epoch_raw[0], epoch_raw[1], epoch_acc[0], epoch_acc[1])
    logging.info(batch_info)
    print(batch_info)


if __name__ == '__main__':
    val()
