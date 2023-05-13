import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.models import resnet34
from torchvision.datasets import EMNIST
import numpy as np
import time
from torch.utils.data import ConcatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    acc_history = {"train":[], "test":[]}
    loss_history = {"train":[], "test":[]}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_metric = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.cuda()
                labels = labels.cuda()
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):


                    outputs = model(inputs.float())
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                labels = labels.cpu()
                preds = preds.cpu()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            if phase == "test":
                scheduler.step(epoch_loss)

            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            loss_history[phase].append(epoch_loss)
            acc_history[phase].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss,epoch_acc))

            if phase == "test":
                if epoch_acc > best_metric:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_metric = epoch_acc


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_metric))

    model.load_state_dict(best_model_wts)
    return best_model_wts, (acc_history, loss_history)


def find_initial_lr(dataloader, model_, criterion):
  model = copy.deepcopy(model_)
  lr_list = np.logspace(-5, stop = 0, num = len(dataloader))
  losses = []
  model.train()
  with torch.set_grad_enabled(True):
    for batch, (X, y) in enumerate(dataloader):
      X = X.cuda()
      y = y.cuda()
      outputs = model(X.float())
      loss = criterion(outputs, y)
      optimizer = torch.optim.AdamW(model.parameters(), lr=lr_list[batch])
      losses.append(loss.item())
      loss.backward()
      optimizer.step()
  best_ind = np.asarray(losses).argmin()
  best_lr = lr_list[best_ind]
  return best_lr

def get_dataloaders():
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=0, std=255.0, inplace=True)])
    augment_transform = torchvision.transforms.Compose(
        [torchvision.transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=(10, 10, 10, 10)),
         torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=0, std=255.0, inplace=True)])

    RawTrain = EMNIST(split="balanced", download=True, root="./", train=True, transform=transform)
    RawTest = EMNIST(split="balanced", download=True, root="./", train=False, transform=transform)
    AugmentedTrain = EMNIST(split="balanced", download=True, root="./", train=True, transform=augment_transform)

    lst = list(range(36, 47))
    lst.append(18)
    lst.append(24)
    lst.append(26)
    classes = torch.tensor(lst)

    indices_train = (torch.tensor(RawTrain.targets)[..., None] == classes).any(-1).nonzero(as_tuple=True)[0].numpy()

    indices_test = (torch.tensor(RawTest.targets)[..., None] == classes).any(-1).nonzero(as_tuple=True)[0].numpy()

    RawTrain.data = np.delete(RawTrain.data, indices_train, axis=0)
    RawTrain.classes = [RawTrain.classes[i] for i in range(len(RawTrain.classes)) if i not in lst]
    RawTrain.targets = np.delete(RawTrain.targets, indices_train, axis=0)
    RawTrain.targets[(19 <= RawTrain.targets) & (RawTrain.targets <= 24)] -= 1
    RawTrain.targets[RawTrain.targets == 25] = 23
    RawTrain.targets[RawTrain.targets >= 27] -= 3

    RawTest.data = np.delete(RawTest.data, indices_test, axis=0)
    RawTest.classes = [RawTest.classes[i] for i in range(len(RawTest.classes)) if i not in lst]
    RawTest.targets = np.delete(RawTest.targets, indices_test, axis=0)
    RawTest.targets[(19 <= RawTest.targets) & (RawTest.targets <= 24)] -= 1
    RawTest.targets[RawTest.targets == 25] = 23
    RawTest.targets[RawTest.targets >= 27] -= 3

    AugmentedTrain.data = np.delete(AugmentedTrain.data, indices_train, axis=0)
    AugmentedTrain.classes = [AugmentedTrain.classes[i] for i in range(len(AugmentedTrain.classes)) if i not in lst]
    AugmentedTrain.targets = np.delete(AugmentedTrain.targets, indices_train, axis=0)
    AugmentedTrain.targets[(19 <= AugmentedTrain.targets) & (AugmentedTrain.targets <= 24)] -= 1
    AugmentedTrain.targets[AugmentedTrain.targets == 25] = 23
    AugmentedTrain.targets[AugmentedTrain.targets >= 27] -= 3

    TrainDataset = ConcatDataset([RawTrain, AugmentedTrain])

    batch_size = 512
    TrainLoader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    TestLoader = DataLoader(RawTest, batch_size=batch_size)
    DataLoaders = {"train": TrainLoader, "test": TestLoader}

    return DataLoaders

def main():
    model = resnet34(pretrained=False)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.relu = nn.PReLU(64)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(512, 33)
    )
    model.cuda()

    dataloaders = get_dataloaders()

    criterion = nn.CrossEntropyLoss()

    best_in_lr = find_initial_lr(dataloaders["train"], model, criterion)

    optimizer = torch.optim.AdamW(model.parameters(), lr=best_in_lr)

    best_weights, histories = train_model(model, dataloaders, criterion, optimizer, num_epochs=25)

    torch.save(best_weights, "model.pth")


