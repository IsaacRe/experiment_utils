from os.path import join
from .argument_parsing import *
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn
import torch.optim
from torchvision.models import resnet18, resnet34
from .dataset import get_dataloader, get_dataloader_incr, JointDataLoader, get_subset_dataloaders


model_factories = {
    'resnet18': resnet18,
    'resnet34': resnet34
}


def save_model(model, save_path, device=0, state_dict=True):
    model.cpu()
    if state_dict:
        torch.save(model.state_dict(), save_path)
    else:
        torch.save(model, save_path)
    if device != 'cpu':
        model.cuda(device)


def test(model, loader, device=0, multihead=False, debug=False):
    active_outputs = np.arange(model.fc.out_features)
    if hasattr(model, 'active_outputs'):
        active_outputs = np.array(model.active_outputs)

    # if multihead, set all classes other than those of current exposure to inactive (outputs wont contribute)
    if multihead:
        classes = np.array(loader.classes)
    # otherwise, make only classes that the model hasnt been exposed to yet inactive (retain previously trained outputs)
    else:
        classes = active_outputs

    # TODO shorten class_idxs
    num_classes = model.fc.out_features
    total, correct = np.zeros(num_classes) + 1e-20, np.zeros(num_classes)
    class_idxs = np.arange(num_classes)[None].repeat(loader.batch_size, axis=0)

    inactive_classes = np.where(
        (np.arange(model.fc.out_features)[:, None].repeat(len(classes), 1) == classes[None]).sum(axis=1) == 0
    )

    with torch.no_grad():
        for i, x, y in tqdm(loader):
            x, y = x.to(device), y.to(device)
            out = model(x)

            out[:, inactive_classes] = float('-inf')
            pred = out.argmax(dim=1)

            # TODO debug
            y, pred = (y.cpu().numpy()[:, None] == class_idxs[:len(y)]), \
                      (pred.cpu().numpy()[:, None] == class_idxs[:len(y)])
            total += y.sum(axis=0)
            correct += np.logical_and(pred, y).sum(axis=0)

        print('%d/%d (%.2f%%)' % (correct.sum(), total.sum(), correct.sum() / total.sum() * 100.))
        return correct, total


def test_all(model, loaders, device=0, multihead=False):
    correct = total = 0
    for loader in loaders:
        c, t = test(model, loader, device=device, multihead=multihead)
        correct = correct + c
        total = total + t
    return correct, total


def train(args: TrainingArgs, model, train_loader, test_loader, device=0, multihead=False,
          fc_only=False, optimize_modules=None):
    active_outputs = np.arange(model.fc.out_features)
    if hasattr(model, 'active_outputs'):
        active_outputs = np.array(model.active_outputs)

    if fc_only:
        optimize_modules = []
    elif optimize_modules is None:
        optimize_modules = [model]

    params = set()
    for module in optimize_modules:
        params = params.union(set(module.parameters()))
    params = list(params)

    model.train()

    def get_optim(lr):
        if args.adam:
            return torch.optim.Adam(params,
                                    lr=lr,
                                    weight_decay=args.weight_decay)
        return torch.optim.SGD(params,
                               lr=lr,
                               nesterov=args.nesterov,
                               momentum=args.momentum,
                               weight_decay=args.weight_decay)

    def get_scheduler(optim):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    lr = args.lr
    optim = get_optim(lr)
    if args.use_schedule:
        scheduler = get_scheduler(optim)
    loss_fn = torch.nn.CrossEntropyLoss()
    total, correct = [], []
    mean_losses = []
    torch.manual_seed(args.seed)  # seed dataloader shuffling

    # if multihead, set all classes other than those of current exposure to inactive (outputs wont contribute)
    if multihead:
        classes = np.array(train_loader.classes)
    # otherwise, make only classes that the model hasnt been exposed to yet inactive (retain previously trained outputs)
    else:
        classes = active_outputs

    inactive_classes = np.where(
        (np.arange(model.fc.out_features)[:, None].repeat(len(classes), 1) == classes[None]).sum(axis=1) == 0
    )

    for e in range(args.epochs):
        # check for lr decay
        if args.use_schedule:
            scheduler.step()
        elif e in args.decay_epochs:
            lr /= args.lr_decay
            optim = get_optim(lr)

        print('Beginning epoch %d/%d' % (e + 1, args.epochs))
        losses = []

        for i, x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            out = model(x)

            out[:, inactive_classes] = float('-inf')

            loss = loss_fn(out, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            losses += [loss.item()]

        mean_loss = sum(losses) / len(losses)
        print('Mean loss for epoch %d: %.4f' % (e, mean_loss))
        print('Test accuracy for epoch %d:' % e, end=' ')

        mean_losses += [mean_loss]

        model.eval()
        correct_, total_ = test(model, test_loader, device=device, multihead=multihead)
        model.train()
        total += [total_]
        correct += [correct_]
        if args.save_acc:
            np.savez(join(args.acc_save_dir, args.acc_save_path),
                     train_loss=np.array(mean_losses),
                     val_accuracy=np.stack(correct, axis=0) / np.stack(total, axis=0))
        save_model(model, join(args.model_save_dir, args.model_save_path), device=device)


def train_batch_multihead(args: TrainingArgs, model, train_loaders, test_loaders, device=0, fc_only=False):
    num_batches = len(train_loaders[0])
    assert all([len(l) == num_batches for l in train_loaders]), 'some train_loaders have different number of batches'
    num_tasks = len(train_loaders)
    batch_size = train_loaders[0].batch_size
    assert all([l.batch_size == batch_size for l in train_loaders]), 'some train_loaders have different batch size'
    classes_per_task = len(train_loaders[0].classes)
    assert all([len(l.classes) == classes_per_task for l in train_loaders]),\
        'some train_loaders have different num classes'
    total_classes = classes_per_task * num_tasks

    model.train()
    def get_optim(lr):
        params = model.fc.parameters() if fc_only else model.parameters()
        return torch.optim.SGD(params,
                               lr=lr,
                               nesterov=args.nesterov,
                               momentum=args.momentum,
                               weight_decay=args.weight_decay)
    lr = args.lr
    optim = get_optim(lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    total, correct = [], []
    mean_losses = []
    torch.manual_seed(args.seed)  # seed dataloader shuffling

    # assuming same number of classes per loader
    classes = np.array([l.classes for l in train_loaders])  # [num_tasks X classes_per_task]
    classes = classes[:,None].repeat(batch_size, 1).reshape(-1, classes_per_task)  # [num_tasks * batch_size X classes_per_task]

    x_idxs = np.arange(batch_size * num_tasks)[:, None].repeat(classes_per_task, 1).reshape(-1)  # [batch_size * num_tasks * classes_per_task]
    y_idxs = classes.reshape(-1)  # [batch_size * num_tasks * classes_per_task]

    inactive_class_mask = np.ones((num_tasks * batch_size, total_classes)).astype(np.bool_)  # [num_tasks * batch_size X total_classes]
    inactive_class_mask[(x_idxs, y_idxs)] = False

    joint_trainloader = JointDataLoader(*train_loaders)

    for e in range(args.epochs):
        # check for lr decay
        if e in args.decay_epochs:
            lr /= args.lr_decay
            optim = get_optim(lr)

        print('Beginning epoch %d/%d' % (e + 1, args.epochs))
        losses = []

        for i, x, y in tqdm(joint_trainloader):
            x, y = x.to(device), y.to(device)
            out = model(x)

            out[inactive_class_mask] = float('-inf')

            loss = loss_fn(out, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            losses += [loss.item()]

        mean_loss = sum(losses) / len(losses)
        print('Mean loss for epoch %d: %.4f' % (e, mean_loss))
        print('Test accuracy for epoch %d:' % e, end=' ')

        mean_losses += [mean_loss]

        model.eval()
        correct_, total_ = test_all(model, test_loaders, device=device, multihead=True)
        model.train()
        total += [total_]
        correct += [correct_]
        if args.save_acc:
            np.savez(join(args.acc_save_dir, args.acc_save_path),
                     train_loss=np.array(mean_losses),
                     val_accuracy=np.stack(correct, axis=0) / np.stack(total, axis=0))
        save_model(model, join(args.model_save_dir, args.model_save_path), device=device)


def initialize_model(args: ModelInitArgs, device=0):
    # load network
    torch.manual_seed(args.seed)  # seed random network initialization
    model = model_factories[args.arch](pretrained=args.pretrained)
    model.fc = torch.nn.Linear(model.fc.in_features, args.num_classes, bias=True)
    if device != 'cpu':
        model.cuda(device)
    return model


# load test/train dataloaders
def get_dataloaders(args: DataArgs, load_train=True, load_test=True):
    if load_train:
        train_loader, val_loader = get_dataloader(batch_size_train=args.batch_size_train,
                                                  batch_size_test=args.batch_size_test,
                                                  data_dir=args.data_dir,
                                                  base=args.dataset,
                                                  num_classes=args.num_classes,
                                                  train=True,
                                                  num_workers=args.num_workers,
                                                  pin_memory=args.pin_memory,
                                                  val_ratio=args.val_ratio,
                                                  seed=args.seed)
    else:
        train_loader, val_loader = None, None
    if load_test:
        test_loader = get_dataloader(batch_size_train=args.batch_size_test,
                                     data_dir=args.data_dir,
                                     base=args.dataset,
                                     num_classes=args.num_classes,
                                     train=False,
                                     num_workers=args.num_workers,
                                     pin_memory=args.pin_memory)
    else:
        test_loader = None
    return train_loader, val_loader, test_loader


def get_subset_data_loaders(args: DataArgs, num_samples):
    return get_subset_dataloaders(num_samples, num_samples,
                                  batch_size=args.batch_size_train,
                                  batch_size_test=args.batch_size_test,
                                  data_dir=args.data_dir,
                                  base=args.dataset,
                                  num_classes=args.num_classes,
                                  train=True,
                                  num_workers=args.num_workers,
                                  pin_memory=args.pin_memory,
                                  val_ratio=args.val_ratio,
                                  seed=args.seed)


def get_dataloaders_incr(args: IncrDataArgs, load_train=True, load_test=True, multihead_batch=False):
    if args.exposure_class_splits is not None:
        class_splits = args.exposure_class_splits
        classes_per_exp = args.classes_per_exposure
        assert len(args.exposure_class_splits) % args.classes_per_exposure == 0,\
            'classes_per_exposure must agree with exposure_class_splits'
        args.exposure_class_splits = [class_splits[i:i+classes_per_exp] for i in range(0,
                                                                                       len(class_splits),
                                                                                       classes_per_exp)]
        # TODO debug exposure_class_splits
    if load_train:
        train_loaders, val_loaders = get_dataloader_incr(batch_size_train=args.batch_size_train,
                                                         batch_size_test=args.batch_size_test,
                                                         data_dir=args.data_dir,
                                                         base=args.dataset,
                                                         num_classes=args.num_classes,
                                                         train=True,
                                                         num_workers=args.num_workers,
                                                         pin_memory=args.pin_memory,
                                                         val_ratio=args.val_ratio,
                                                         val_idxs_path=args.val_idxs_path,
                                                         train_idxs_path=args.train_idxs_path,
                                                         seed=args.seed,
                                                         classes_per_exposure=args.classes_per_exposure,
                                                         exposure_class_splits=args.exposure_class_splits,
                                                         scale_batch_size=multihead_batch)
    else:
        train_loaders, val_loaders = None, None
    if load_test:
        test_loaders = get_dataloader_incr(batch_size_test=args.batch_size_test,
                                           data_dir=args.data_dir,
                                           base=args.dataset,
                                           num_classes=args.num_classes,
                                           train=False,
                                           num_workers=args.num_workers,
                                           pin_memory=args.pin_memory,
                                           classes_per_exposure=args.classes_per_exposure,
                                           exposure_class_splits=args.exposure_class_splits,
                                           scale_batch_size=multihead_batch)
    else:
        test_loaders = None
    return train_loaders, val_loaders, test_loaders


if __name__ == '__main__':
    model_args, save_init_args, data_args, train_args = parse_args(ModelInitArgs, InitModelPath, DataArgs, TrainRefModelArgs)
    model = initialize_model(model_args, device=0)
    save_model(model, save_init_args.init_model_path)
    train_loader, val, test_loader = get_dataloaders(data_args)
    train(train_args, model, train_loader, test_loader, device=0)
