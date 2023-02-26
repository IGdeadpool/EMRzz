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
    x_train = np.array(input_file['train_set/trace'], dtype = 'f4')
    y_train = np.array(input_file['train_set/labels'],dtype = 'f4')
    print(x_train.shape)

    x_valid = np.array(input_file['valid_set/trace'], dtype = 'f4')
    y_valid = np.array(input_file['valid_set/labels'], dtype ='f4')
    print(x_valid.shape)
    print(x_train.shape, y_train.shape)
    x_train = torch.from_numpy(x_train)

    # print(x_train)
    y_train = torch.from_numpy(y_train)

    x_valid = torch.from_numpy(x_valid)


    y_valid = torch.from_numpy(y_valid)

    # normalize



    train_dataset = Data.TensorDataset(x_train, y_train)
    valid_dataset = Data.TensorDataset(x_valid, y_valid)

    del y_train
    del x_train
    del y_valid
    del x_valid

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)


    return train_loader, valid_loader

class KeyRecovery(nn.Module):
    def __init__(self):
        super(KeyRecovery, self).__init__()
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 4, 2),
            nn.AvgPool1d(1),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(4, 8, 2),
            nn.AvgPool1d(1),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(8, 16, 2),
            nn.AvgPool1d(1),
            nn.LeakyReLU()
        )
        self.fc1 = nn.Linear(1552, 4096)
        self.fc2 = nn.Linear(4096, 256)
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
        x = torch.reshape(x, (-1,256))
        x = self.drop(x)
        x = F.softmax(x, dim=0)
        # x = F.sigmoid(x)
        return x

def to_categorical(y, num_classes):
    y = y.numpy().astype(int)
    y = np.eye(num_classes, dtype='uint16')[y]
    y= y.astype(np.float32, order='C', casting='unsafe')
    y = torch.from_numpy(y)

    return y

def partial_correct_accuracy(y_true ,batch_size,y_pred):
    accuracy = 0.0
    for i in range(batch_size):
        accuracy+=y_pred[i][int(y_true[i])]



    return accuracy

if __name__ == "__main__":
    if len(sys.argv)!=2:
        dataset = "AES_key_recover_train.hdf5"
        epoches = 100
        batch_size = 10
        model_file = "saved_model.txt"
    else:

        # todo: read parameters
        dataset = "AES_key_recover_train.hdf5"
        epoches = 100
        batch_size = 10
        model_file = "saved_model.txt"

    train_loader, valid_loader = load_dataset(dataset, batch_size)
    print("data finished")

    model_parameter = {}
    subkey_index = [0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15]
    for k in subkey_index:
        net = KeyRecovery()
        net.to(device)
        learning_rate = 0.1
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        loss_function = nn.CrossEntropyLoss()
        scheduler = StepLR(optimizer, 10, gamma=0.5)
        for epoch in range(epoches):
            print("runing epoch {}".format(epoch))
            # training
            accuracy_train = 0.0
            loss_train = 0.0
            net.train()
            for step, (x_train, y_train) in enumerate(train_loader):

                x_train = x_train.to(device)
                x_train = x_train.unsqueeze(1)
                # modelsize(net, x_train)
                # y_train = y_train.squeeze()
                y_train_value = np.zeros((batch_size))
                for i in range(batch_size):
                    y_train_value[i] = y_train[i][k]
                y_train_value = torch.from_numpy(y_train_value)
                y_train_value = y_train_value.to(device)
                output = net(x_train)
                loss = loss_function(output, y_train_value.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                output_arr = output.cpu().detach().numpy()
                accuracy_train += partial_correct_accuracy(y_train_value.cpu().numpy(), batch_size, output_arr)
                loss_train += loss.data.cpu().numpy()
                if step % 10 ==9:
                    print('subkey_order: ', k, 'Epoch: ', epoch, 'Batch: ', step,
                          '| train loss: %.4f' % (loss_train / batch_size/10),
                          '|train accuracy: %.4f' % (accuracy_train / batch_size/10),
                          '|learning rate: %.6f' % optimizer.param_groups[0]['lr'])
                    accuracy_train = 0.0
                    loss_train = 0.0
            loss_valid = 0.0
            accuracy_valid = 0.0
            net.eval()
            for i, (x_valid, y_valid) in enumerate(valid_loader):
                x_valid = x_valid.to(device)
                x_valid = x_valid.unsqueeze(1)
                output = net(x_valid)
                y_valid_value = np.zeros((batch_size))
                for i in range(batch_size):
                    y_valid_value[i] = y_valid[i][k]
                y_valid_value = torch.from_numpy(y_valid_value)
                y_valid_value = y_valid_value.to(device)
                loss = loss_function(output, y_valid_value.long())
                # print(output, y_valid_one_hot)
                loss_valid += loss.data.cpu().numpy()

                output_arr = output.cpu().detach().numpy()
                accuracy_valid += partial_correct_accuracy(y_valid_value.cpu().numpy(), batch_size, output_arr)
            print('Epoch: ', epoch, '| validation loss: %.4f' % (loss_valid / 200),
                  '|validatiton accuracy: %.4f' % (accuracy_valid / 200))

            scheduler.step()
            torch.cuda.empty_cache()
        model_name = "model" + str(k)
        optimizer_name = "optimizer" + str(k)
        model_parameter[model_name] = net.state_dict()
        model_parameter[optimizer_name] = optimizer.state_dict()

    torch.save(model_parameter, model_file)