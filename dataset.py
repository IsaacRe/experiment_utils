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


def get_dataset(data_dir='../data', base='CIFAR100', num_classes=100, train=True, download=False,
                **dataset_kwargs):
    if base == 'CIFAR10':
        assert num_classes <= 10, 'CIFAR10 only has 10 classes'
    elif base == 'CIFAR100':
        assert num_classes <= 100, 'CIFAR100 only has 100 classes'
    ExtendedDataset = extend_dataset(base)
    dataset = ExtendedDataset(data_dir, num_classes=num_classes, train=train, download=download,
                              **dataset_kwargs)
    return dataset


def get_dataloader(batch_size_train=100, batch_size_test=200, data_dir='../data', base='CIFAR100', num_classes=100,
                   train=True, download=False, val_ratio=0.01, num_workers=4, pin_memory=False, seed=1, **dataset_kwargs):
    dataset = get_dataset(data_dir=data_dir, base=base, num_classes=num_classes, train=train,
                          download=download, **dataset_kwargs)

    if train:
        np.random.seed(seed)
        data_idxs = np.arange(len(dataset))
        np.random.shuffle(data_idxs)
        val_size = int(len(dataset) * val_ratio)
        train_sampler, val_sampler = SubsetRandomSampler(data_idxs[val_size:]), \
                                     SubsetRandomSampler(data_idxs[:val_size])
        train_loader = DataLoader(dataset,
                                  batch_size=batch_size_train,
                                  sampler=train_sampler,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
        val_loader = DataLoader(dataset,
                                batch_size=batch_size_test,
                                sampler=val_sampler,
                                num_workers=num_workers,
                                pin_memory=pin_memory)
        train_loader.classes = val_loader.classes = list(range(num_classes))

        return train_loader, val_loader
    else:
        test_loader = DataLoader(dataset,
                                 batch_size=batch_size_test,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)
        test_loader.classes = list(range(num_classes))

        return test_loader


def get_subset_dataloaders(*num_samples, batch_size=100, batch_size_test=100, data_dir='../data', base='CIFAR10',
                           num_classes=10, train=True, download=False,
                           val_ratio=0.01, num_workers=4, pin_memory=False, seed=1, disjoint=True,
                           **dataset_kwargs):
    dataset = get_dataset(data_dir=data_dir, base=base, num_classes=num_classes, train=train,
                          download=download, **dataset_kwargs)

    np.random.seed(seed)
    data_idxs = np.arange(len(dataset))
    np.random.shuffle(data_idxs)
    val_size = int(len(dataset) * val_ratio)
    val_sampler = SubsetRandomSampler(data_idxs[:val_size])
    val_loader = DataLoader(dataset, batch_size=batch_size_test, sampler=val_sampler, num_workers=num_workers,
                            pin_memory=pin_memory)

    train_loaders = []
    samples_processed = val_size
    for n_samples in num_samples:
        sampler = SubsetRandomSampler(data_idxs[samples_processed:samples_processed+n_samples])
        samples_processed += n_samples
        train_loaders += [DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                                     pin_memory=pin_memory)]

    return train_loaders, val_loader


def get_dataloader_incr(batch_size_train=100, batch_size_test=200, data_dir='../data', base='CIFAR100', num_classes=100,
                        train=True, download=False, val_ratio=0.01, num_workers=4, pin_memory=False, seed=1,
                        classes_per_exposure=10, exposure_class_splits=None, scale_batch_size=False,
                        val_idxs_path=None, train_idxs_path=None, **dataset_kwargs):
    train_val_split = None
    if val_idxs_path is not None:
        assert train_idxs_path is not None, 'must specify both val and train indices'
        train_val_split = np.load(train_idxs_path), np.load(val_idxs_path)

    dataset = get_dataset(data_dir=data_dir, base=base, num_classes=num_classes, train=train,
                          download=download, train_val_split=train_val_split, **dataset_kwargs)

    if exposure_class_splits is None:
        assert num_classes % classes_per_exposure == 0, "specified classes per exposure (%d) does not evenly divide " \
                                                        "specified number of classes (%d)" % (classes_per_exposure,
                                                                                              num_classes)
        exposure_class_splits = [list(range(c, c + classes_per_exposure))
                                 for c in range(0, num_classes, classes_per_exposure)]

    if scale_batch_size:
        # scale down batch size by the number of total loader if we will be loading data across loaders concurrently
        batch_size_train = batch_size_train // len(exposure_class_splits)

    targets = np.array(dataset.targets)

    if train:
        train_loaders = []
        val_loaders = []

        np.random.seed(seed)

        for classes in exposure_class_splits:

            if val_idxs_path is not None:
                val_idxs = dataset.val_indices
                train_idxs = dataset.train_indices
                val_idxs_by_class = []
                train_idxs_by_class = []
                for c in classes:
                    val_mask = targets[val_idxs] == c
                    train_mask = targets[train_idxs] == c
                    val_idxs_by_class += [val_idxs[val_mask]]
                    train_idxs_by_class += [train_idxs[train_mask]]

                train_sampler = SubsetRandomSampler(np.concatenate(train_idxs_by_class))
                val_sampler = SubsetRandomSampler(np.concatenate(val_idxs_by_class))
            else:
                idxs_by_class = []
                val_sizes = []
                for c in classes:
                    c_idxs = np.where(targets == c)[0]
                    np.random.shuffle(c_idxs)
                    idxs_by_class += [c_idxs]
                    val_sizes += [int(len(c_idxs) * val_ratio)]

                train_sampler = SubsetRandomSampler(np.concatenate([c_idxs[val_size:] for c_idxs, val_size
                                                                    in zip(idxs_by_class, val_sizes)]))
                val_sampler = SubsetRandomSampler(np.concatenate([c_idxs[:val_size] for c_idxs, val_size
                                                                  in zip(idxs_by_class, val_sizes)]))

            train_loader = DataLoader(dataset,
                                      batch_size=batch_size_train,
                                      sampler=train_sampler,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory)
            val_loader = DataLoader(dataset,
                                    batch_size=batch_size_test,
                                    sampler=val_sampler,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory)
            train_loader.classes = val_loader.classes = classes

            train_loaders += [train_loader]
            val_loaders += [val_loader]

        return train_loaders, val_loaders
    else:
        test_loaders = []

        for classes in exposure_class_splits:
            exposure_idxs = np.concatenate([np.where(targets == c)[0] for c in classes])

            exposure_loader = DataLoader(
                dataset,
                batch_size=batch_size_test,
                sampler=SubsetRandomSampler(exposure_idxs),
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            exposure_loader.classes = classes
            test_loaders += [exposure_loader]

        return test_loaders


def extend_dataset(base_dataset):
    assert base_dataset in DATASETS, "Dataset '%s' is not in list of accepted datasets for extension" % base_dataset

    class ExtendedDataset(DATASETS[base_dataset]):

        def __init__(self, *args, num_classes=100, keep_index=True, img_size=224, train=True,
                     load_features_path=None,
                     save_features_path=None,
                     extract_at_layer=None,
                     feature_extractor=None,
                     train_val_split=None,
                     **kwargs):
            super(ExtendedDataset, self).__init__(*args, train=train, **kwargs)
            self.train = train
            self.num_classes = num_classes
            self.train_indices, self.val_indices = None, None
            if train_val_split is not None:
                assert sum([len(split) for split in train_val_split]) == len(self), \
                    'loaded train-val split is incompatible with the dataset loaded'
                self.train_indices, self.val_indices = train_val_split
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

        def _update_train_val_split(self, mask):
            train_mask, val_mask = np.zeros(mask.shape), np.zeros(mask.shape).astype(np.bool_)
            train_mask[self.train_indices] = True
            val_mask[self.val_indices] = True
            train_mask = np.logical_and(mask, train_mask)
            val_mask = np.logical_and(mask, val_mask)
            self.train_indices, = np.where(train_mask[mask])
            self.val_indices, = np.where(val_mask[mask])

        def _set_data(self):
            label_arr = np.array(self.targets)
            mask = self._get_data_mask(label_arr)
            if self.train_indices is not None:
                self._update_train_val_split(mask)
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


class JointDataLoader:

    def __init__(self, *dataloaders: DataLoader):
        self.loaders = dataloaders

    def __iter__(self):
        for zipped_batch in zip(*self.loaders):
            batch_items = zip(*zipped_batch)
            yield (torch.cat(i) for i in batch_items)

    def __len__(self):
        return min([len(l) for l in self.loaders])
