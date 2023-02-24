import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import os
import h5py
import sys
import torch.utils.data as Data
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
root =os.getcwd() + '/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def check_file_exist(file_path):
    if os.path.exists(file_path) == False:
        print("Error: provided file path does not exist")
        sys.exit(-1)
    return
def load_dataset(dataset, batch_size):
    check_file_exist(dataset)
    try:
        input_file = h5py.File(dataset, "r")
    except:
        print("Error: can't open HDF5 file for reading")
        sys.exit(-1)

    # batchnorm2d = nn.BatchNorm2d(1)
    x_train = np.array(input_file['train_set/trace'], dtype = np.int16)
    y_train = np.array(input_file['train_set/labels'])

    x_valid = np.array(input_file['valid_set/trace'], dtype=np.int16)
    y_valid = np.array(input_file['valid_set/labels'])
    print(x_train.shape, y_train.shape)
    x_train = x_train.astype(np.float32,order='C', casting= 'unsafe')
    x_train = torch.from_numpy(x_train)
    x_train = x_train.unsqueeze(1)
    # print(x_train)
    # x_train = batchnorm2d(x_train)
    y_train = y_train.astype(np.float32, order='C', casting='unsafe')
    y_train = torch.from_numpy(y_train)

    x_valid= x_valid.astype(np.float32, order='C', casting='unsafe')
    x_valid = torch.from_numpy(x_valid)
    x_valid = x_valid.unsqueeze(1)
    # x_valid =batchnorm2d(x_valid)
    y_valid = y_valid.astype(np.float32, order='C', casting='unsafe')
    y_valid = torch.from_numpy(y_valid)

    # normalize
    x_train /= 255
    x_valid /= 255



    train_dataset = Data.TensorDataset(x_train, y_train)
    valid_dataset = Data.TensorDataset(x_valid, y_valid)


    """    loader = Data.DataLoader(train_dataset, batch_size=len(train_dataset), num_workers=1)
    data = next(iter(loader))
    train_mean = data[0].mean()
    train_std = data[0].std()
    x_train = (x_train-train_mean)/train_std
    train_dataset = Data.TensorDataset(x_train, y_train)"""

    del y_train
    del x_train
    del y_valid
    del x_valid

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)


    return train_loader, valid_loader

def modelsize(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums

    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size * 2 / 1000 / 1000))

class KeyRecovery(nn.Module):
    def __init__(self):
        super(KeyRecovery, self).__init__()
        self.bn1 = nn.BatchNorm1d(8192)
        self.bn2 = nn.BatchNorm1d(4096)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size =5, stride=2, padding =2),
            nn.BatchNorm2d(4, affine= True),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 16, 5, 2, 2),
            nn.BatchNorm2d(16, affine=True),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 4, 5, 2, 2),
            nn.BatchNorm2d(4, affine=True),
            nn.LeakyReLU()
        )
        self.fc1 = nn.Linear(15876,8192)
        self.fc2 = nn.Linear(8192, 4096)
        self.drop = nn.Dropout(p=0.2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.reshape(x, (-1,16,256))
        x = self.drop(x)
        x = F.log_softmax(x, dim=2)
        # x = F.sigmoid(x)
        return x

def to_categorical(y, num_classes):
    y = y.numpy().astype(int)
    y = np.eye(num_classes, dtype='uint16')[y]
    y= y.astype(np.float32, order='C', casting='unsafe')
    y = torch.from_numpy(y)

    return y

def partial_correct_accuracy(y_true ,y_pred):
    num_correct =0
    num_wrong =0

    for i in range(len(y_true)):
        for n in range(16):
            target = np.argmax(y_true[i][n])
            index = np.argmax(y_pred[i][n])
            if target == index:
                num_correct+=1
            else:
                num_wrong+=1

    accuracy = num_correct/ (num_wrong+num_correct)



    return accuracy

if __name__ == "__main__":
    if len(sys.argv)!=2:
        dataset = "AES_key_recover_train_1000.hdf5"
        epoches = 100
        batch_size = 10
        model_file = "saved_model.txt"
    else:

        # todo: read parameters
        dataset = "AES_key_recover_train_1000.hdf5"
        epoches = 100
        batch_size = 10
        model_file = "saved_model.txt"

    train_loader, valid_loader = load_dataset(dataset, batch_size)
    print("data finished")
    net = KeyRecovery()
    net.to(device)

    learning_rate = 0.1
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_function = nn.NLLLoss()
    scheduler = StepLR(optimizer, 10, gamma=0.5)
    for epoch in range(epoches):
        print("runing epoch {}".format(epoch))
        # training
        accuracy_train = 0.0
        loss_train = 0.0
        net.train()
        for step, (x_train, y_train) in enumerate(train_loader):

            x_train = x_train.to(device)
            y_train_one_hot = to_categorical(y_train, num_classes=256)
            y_train_one_hot = y_train_one_hot.to(device)
            # modelsize(net, x_train)
            # y_train = y_train.squeeze()

            output = net(x_train)
            # loss = loss_function(output, y_train)
            loss = -torch.sum(output*y_train_one_hot)/batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            output_arr = output.cpu().detach().numpy()
            accuracy_train += partial_correct_accuracy(y_train_one_hot.cpu().numpy(), output_arr)
            loss_train += loss.data.cpu().numpy()

            if step % 10 == 9:
                print('Epoch: ', epoch, 'Batch: ', step+1, '| train loss: %.4f' % (loss_train/10),
                      '|train accuracy: %.4f' % (accuracy_train/10),
                      '|learning rate: %.6f' % optimizer.param_groups[0]['lr'])
                accuracy_train = 0.0
                loss_train = 0.0

        # net.eval()
        loss_valid = 0.0
        accuracy_valid = 0.0
        for i, (x_valid, y_valid) in enumerate(valid_loader):

            x_valid = x_valid.to(device)
            y_valid_one_hot = to_categorical(y_valid, num_classes=256)
            y_valid_one_hot = y_valid_one_hot.to(device)
            output = net.forward(x_valid)

            # loss = loss_function(output, y_valid)
            # print(output, y_valid_one_hot)
            loss = -torch.sum(output * y_valid_one_hot) / batch_size
            loss_valid += loss.data.cpu().numpy()

            output_arr = output.cpu().detach().numpy()
            accuracy_valid += partial_correct_accuracy(y_valid_one_hot.cpu().numpy(), output_arr)
        print('Epoch: ', epoch, '| validation loss: %.4f' % (loss_valid/30) ,
                 '|validatiton accuracy: %.4f' % (accuracy_valid/30))

        scheduler.step()
        torch.cuda.empty_cache()

    torch.save(net.state_dict(), model_file)