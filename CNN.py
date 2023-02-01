import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
import time
import pandas as pd
import json
from IPython.display import clear_output, display
root =os.getcwd() + '/'
def default_loader(path):
    return Image.open(path).convert('RGB')
class Mydataset(Dataset):
    def __init__(self, txt_name, transform=None, target_tranform=None, loader=default_loader, identify=None):
        super(Mydataset, self).__init__()
        imgs =[]
        fh = open(txt_name, 'r')
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip('\n')
            words = line.split()
            imgs.append((words[0], int(words[1],2)))
        self.img = imgs
        self.transform = transform
        self.target_transform = target_tranform
        self.loader = loader

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        path, label = self.img[index]

        img = self.loader(path)  # 读取出来维度为3， 22， 14 ？？ 无法正常显示图片
        img = np.array(img, dtype=np.float32)
        if self.transform:
            img = self.transform(img)  # Totensor维度改变 torch.Size([3, 22, 14])

        return img, label

train_data = Mydataset(root+'train_set.txt', transforms = transforms.ToTensor())
test_data = Mydataset(root+'test_set.txt', transforms = transforms.ToTensor())

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=True)


class KeyRecovery(nn.Module):
    def __init__(self):
        super(KeyRecovery, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.conv2 = nn.Conv2d(20, 40, 5)
        self.conv3 = nn.Conv2d(40, 80, 5)
        self.fc1 = nn.Linear(80 * 2 * 2, 100)
        self.fc2 = nn.Linear(100, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return F.log_softmax(x)


net = KeyRecovery()
print(net)

loss_func = nn.CrossEntropyLoss()
optim = torch.optim.Adam(net.parameters(), lr=0.01)

# put all hyper params into a OrderedDict, easily expandable
from collections import OrderedDict

params = OrderedDict(
    lr=[.01, .001],
    batch_size=[100, 1000],
    shuffle=[True, False]
)
epochs = 3

# import modules to build RunBuilder and RunManager helper classes
from collections import namedtuple
from itertools import product


# Read in the hyper-parameters and return a Run namedtuple containing all the
# combinations of hyper-parameters
class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


# Helper class, help track loss, accuracy, epoch time, run time,
# hyper-parameters etc. Also record to TensorBoard and write into csv, json
class RunManager():
    def __init__(self):

        # tracking every epoch count, loss, accuracy, time
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        # tracking every run count, run data, hyper-params used, time
        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        # record model, loader and TensorBoard
        self.network = None
        self.train_loader = None
        self.tb = None

    # record the count, hyper-param, model, loader of each run
    # record sample images and network graph to TensorBoard
    def begin_run(self, run, net, train_loader):

        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.network = net
        self.train_loader = train_loader
        self.tb = SummaryWriter(comment=f'-{run}')

        images, labels = next(iter(self.train_loader))
        grid = utils.make_grid(images)

        self.tb.add_image('images', grid)
        self.tb.add_graph(self.network, images)

    # when run ends, close TensorBoard, zero epoch count
    def end_run(self):
        self.tb.close()
        self.epoch_count = 0

    # zero epoch count, loss, accuracy,
    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0

    #
    def end_epoch(self):
        # calculate epoch duration and run duration(accumulate)
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        # record epoch loss and accuracy
        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)

        # Record epoch loss and accuracy to TensorBoard
        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

        # Record params to TensorBoard
        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        # Write into 'results' (OrderedDict) for all run related data
        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results["loss"] = loss
        results["accuracy"] = accuracy
        results["epoch duration"] = epoch_duration
        results["run duration"] = run_duration

        # Record hyper-params into 'results'
        for k, v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')

        # display epoch information and show progress
        clear_output(wait=True)
        display(df)

    # accumulate loss of batch into entire epoch loss
    def track_loss(self, loss):
        # multiply batch size so variety of batch sizes can be compared
        self.epoch_loss += loss.item() * self.loader.batch_size

    # accumulate number of corrects of batch into entire epoch num_correct
    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)

    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    # save end results of all runs into csv, json for further analysis
    def save(self, fileName):

        pd.DataFrame.from_dict(
            self.run_data,
            orient='columns',
        ).to_csv(f'{fileName}.csv')

        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)


# train
m = RunManager()

# get all runs from params using RunBuilder class
for run in RunBuilder.get_runs(params):

    # define the network
    network = KeyRecovery()

    # define a optimizer: optim.Adam
    optimizer = optim.Adam(network.parameters(), lr=run.lr)
    loss_func = nn.CrossEntropyLoss()
    # beigin the RunManager
    m.begin_run(run, network, train_loader)
    for epoch in range(epochs):

        m.begin_epoch()
        for batch in train_loader:
            images = batch[0]
            labels = batch[1]

            # network forward for training
            preds = network(images)

            # Compute loss and gradient backpropgation
            loss = loss_func(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m.track_loss(loss)

            m.track_num_correct(preds, labels)

        m.end_epoch()
    m.end_run()
    m.begin_run(run, network, test_loader)
    for epoch in range(epochs):

        m.begin_epoch()
        for batch in test_loader:
            images = batch[0]
            labels = batch[1]

            # network forward for training
            preds = network(images)

            # Compute loss and gradient backpropgation
            loss = loss_func(preds, labels)
            loss.backward()
            m.track_loss(loss)
            m.track_num_correct(preds, labels)

        m.end_epoch()
    m.end_run()

# when all runs are done, save results to files
m.save('results')