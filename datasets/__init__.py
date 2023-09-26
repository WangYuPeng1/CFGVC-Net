from .aircraft_dataset import AircraftDataset
from .bird_dataset import BirdDataset
from .car_dataset import CarDataset
from .dog_dataset import DogDataset
from .nabirds_dataset import NABirdDataset


def getDataset(target_dataset, resize):
    if target_dataset == 'aircraft':
        return AircraftDataset(mode='train', resize=resize), AircraftDataset(mode='val', resize=resize)
    elif target_dataset == 'bird':
        return BirdDataset(mode='train', resize=resize), BirdDataset(mode='val', resize=resize)
    elif target_dataset == 'car':
        return CarDataset(mode='train', resize=resize), CarDataset(mode='val', resize=resize)
    elif target_dataset == 'dog':
        return DogDataset(mode='train', resize=resize), DogDataset(mode='val', resize=resize)
    elif target_dataset == 'nabirds':
        return NABirdDataset(mode='train', resize=resize), BirdTinyDataset(mode='val', resize=resize)
    else:
        raise ValueError('No Dataset {}'.format(target_dataset))
