from os.path import join
from .argument_parsing import *
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn
import torch.optim
from torchvision.models import resnet18, resnet34
from .dataset import get_dataloader, get_dataloader_incr


model_factories = {
    'resnet18': resnet18,
    'resnet34': resnet34
}


def save_model(model, save_path, device=0, state_dict=False):
    model.cpu()
    if state_dict:
        torch.save(model.state_dict(), save_path)
    else:
        torch.save(model, save_path)
    if device != 'cpu':
        model.cuda(device)


def test(model, loader, device=0, multihead=False):
    with torch.no_grad():
        classes = np.array(loader.classes)
        num_classes = model.fc.out_features
        total, correct = np.zeros(num_classes) + 1e-20, np.zeros(num_classes)
        class_idxs = np.arange(num_classes)[None].repeat(loader.batch_size, axis=0)
        for i, x, y in tqdm(loader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            if multihead:
                pred = out[:, np.array(classes)].argmax(dim=1)
            else:
                pred = out.argmax(dim=1)
            # TODO debug
            y, pred = (y.cpu().numpy()[:, None] == class_idxs[:len(y)]), \
                      (pred.cpu().numpy()[:, None] == class_idxs[:len(y)])
            total += y.sum(axis=0)
            correct += np.logical_and(pred, y).sum(axis=0)

        print('%d/%d (%.2f%%)' % (correct.sum(), total.sum(), correct.sum() / total.sum() * 100.))
        return correct, total


def train(args: TrainingArgs, model, train_loader, test_loader, device=0, multihead=False):
    model.train()
    def get_optim(lr):
        return torch.optim.SGD(model.parameters(),
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

    if multihead:
        assert train_loader.classes == test_loader.classes, \
            "passed train and test dataloaders use different class sets: %s != %s" % (str(train_loader.classes),
                                                                                      str(test_loader.classes))
        classes = np.array(train_loader.classes)
        inactive_classes = np.where(
            (np.arange(model.fc.out_features)[:, None].repeat(len(classes), 1) != classes[None]).sum(axis=1) == 0
        )

    for e in range(args.epochs):
        # check for lr decay
        if e in args.decay_epochs:
            lr /= args.lr_decay
            optim = get_optim(lr)

        print('Beginning epoch %d/%d' % (e + 1, args.epochs))
        losses = []

        for i, x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            out = model(x)

            if multihead:
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

        #model.eval()
        correct_, total_ = test(model, test_loader, device=device)
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


def get_dataloaders_incr(args: IncrDataArgs, load_train=True, load_test=True):
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
                                                         seed=args.seed,
                                                         classes_per_exposure=args.classes_per_exposure,
                                                         exposure_class_splits=args.exposure_class_splits)
    else:
        train_loaders, val_loaders = None, None
    if load_test:
        test_loaders = get_dataloader_incr(batch_size=args.batch_size_test,
                                           data_dir=args.data_dir,
                                           base=args.dataset,
                                           num_classes=args.num_classes,
                                           train=False,
                                           num_workers=args.num_workers,
                                           pin_memory=args.pin_memory,
                                           classes_per_exposure=args.classes_per_exposure,
                                           exposure_class_splits=args.exposure_class_splits)
    else:
        test_loaders = None
    return train_loaders, val_loaders, test_loaders


if __name__ == '__main__':
    model_args, save_init_args, data_args, train_args = parse_args(ModelInitArgs, InitModelPath, DataArgs, TrainRefModelArgs)
    model = initialize_model(model_args, device=0)
    save_model(model, save_init_args.init_model_path)
    train_loader, val, test_loader = get_dataloaders(data_args)
    train(train_args, model, train_loader, test_loader, device=0)
