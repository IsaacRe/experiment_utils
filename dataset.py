import os.path as path
from torch.utils.data.dataset import Dataset
from torchvision.datasets import CIFAR100, CIFAR10  #, ImageNet
from torchvision.transforms import Resize, Compose, RandomCrop, RandomHorizontalFlip
from torchvision.transforms.functional import to_tensor
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch
import numpy as np
from PIL import Image
from .utils.model_tracking import ModuleTracker, TrackingProtocol
from .utils.helpers import find_network_modules_by_name
DATASETS = {'CIFAR10': CIFAR10, 'CIFAR100': CIFAR100}


def get_dataset(data_dir='../data', base='CIFAR100', num_classes=100, train=False, download=False,
                **dataset_kwargs):
    if base == 'CIFAR10':
        assert num_classes <= 10, 'CIFAR10 only has 10 classes'
    elif base == 'CIFAR100':
        assert num_classes <= 100, 'CIFAR100 only has 100 classes'
    ExtendedDataset = extend_dataset(base)
    dataset = ExtendedDataset(data_dir, num_classes=num_classes, train=train, download=download,
                              **dataset_kwargs)
    return dataset


def get_dataloader(batch_size=100, data_dir='../data', base='CIFAR100', num_classes=100, train=True, shuffle=True,
                   download=False, val_size=5000, num_workers=4, pin_memory=False, seed=1, **dataset_kwargs):
    dataset = get_dataset(data_dir=data_dir, base=base, num_classes=num_classes, train=train,
                          download=download, **dataset_kwargs)

    if train:
        np.random.seed(seed)
        data_idxs = np.arange(len(dataset))
        np.random.shuffle(data_idxs)
        train_sampler, val_sampler = SubsetRandomSampler(data_idxs[val_size:]), \
                                     SubsetRandomSampler(data_idxs[:val_size])
        train_loader = DataLoader(dataset,
                                  batch_size=batch_size,
                                  sampler=train_sampler,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
        val_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=val_sampler,
                                num_workers=num_workers,
                                pin_memory=pin_memory)
        return train_loader, val_loader
    else:
        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=num_workers,
                          pin_memory=pin_memory)


def extend_dataset(base_dataset):
    assert base_dataset in DATASETS, "Dataset '%s' is not in list of accepted datasets for extension" % base_dataset

    class ExtendedDataset(DATASETS[base_dataset]):

        def __init__(self, *args, num_classes=100, keep_index=True, img_size=224, train=True,
                     load_features_path=None,
                     save_features_path=None,
                     extract_at_layer=None,
                     feature_extractor=None,
                     **kwargs):
            super(ExtendedDataset, self).__init__(*args, train=train, **kwargs)
            self.train = train
            self.num_classes = num_classes
            self.data_index = None
            self.keep_index = keep_index
            if keep_index:
                self.data_index = np.arange(len(self))
            self.mean_image = None
            self.resize = Resize(img_size)
            self.augment = Compose([RandomCrop(img_size, padding=8), RandomHorizontalFlip()])
            self._set_mean_image()
            self._set_data()

            self.layer = extract_at_layer
            self.use_feature_data = False
            self.feature_data_set = False
            if extract_at_layer:
                self.use_feature_data = True
                self._set_feature_data(feature_extractor, extract_at_layer,
                                       load_features_path=load_features_path,
                                       save_features_path=save_features_path)

        def __getitem__(self, index):
            x_arr, y = self.data[index], self.targets[index]

            if self.keep_index:
                index = self.data_index[index]

            if self.feature_data_set:
                return index, torch.Tensor(x_arr), y

            # convert to PIL Image
            img = Image.fromarray(x_arr)

            # resize image
            img = self.resize(img)

            # apply data augmentation
            if self.train and not self.use_feature_data:
                img = self.augment(img)

            # convert to tensor
            x = to_tensor(img)

            return index, x, y

        def _set_mean_image(self):
            mean_image_path = '%s/%s_mean_image.npy' % (self.root, base_dataset)
            if path.exists(mean_image_path):
                mean_image = np.load(mean_image_path)
            else:
                mean_image = np.mean(self.data, axis=0)
                np.save(mean_image_path, mean_image)
            self.mean_image = mean_image.astype(np.uint8)
            self.data -= self.mean_image[None]

        def _get_data_mask(self, label_arr):
            mask = label_arr == 0
            for i in range(1, self.num_classes):
                mask = np.logical_or(mask, label_arr == i)
            return mask

        def _set_data(self):
            label_arr = np.array(self.targets)
            mask = self._get_data_mask(label_arr)
            self.data = self.data[mask]
            self.targets = list(label_arr[mask])
            if self.keep_index:
                self.data_index = self.data_index[mask]

        def _extract_features(self, feature_extractor, layer, device=0, batch_size=100):
            loader = DataLoader(self, batch_size=batch_size, shuffle=self.train, num_workers=4, pin_memory=True)
            [module] = find_network_modules_by_name(feature_extractor, [layer])
            tracker = ModuleTracker(TrackingProtocol('out'),
                                    **{layer: module})
            feature_extractor.eval()
            return tracker.aggregate_vars(loader, network=feature_extractor, device=device)[layer]['out'].numpy()

        def _set_feature_data(self, feature_extractor, layer, load_features_path, save_features_path):
            if load_features_path:
                self.data = np.load(load_features_path)
            else:
                self.data = self._extract_features(feature_extractor, layer)
                if save_features_path:
                    np.save('%s/%s' % (self.root, save_features_path), self.data)
            self.feature_data_set = True

    return ExtendedDataset
