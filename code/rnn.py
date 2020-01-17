from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from random import shuffle
from lib import *
import random as rd

import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

use_cuda = False

print("Loading Common Words and Creating Features List")
features_list = ['nested_count', 'reply_count','certainty_count', 'extremity_count', 'lexical_diversity_rounded', 'char_count_rounded', 'link_count', 'quote_count',
                'questions_count', 'bold_count', 'avgSentences_count', 'enumeration', 'excla'] #+ common_words

print('Reading File and Creating Data')
df = pd.read_csv('/home/shared/CMV/Delta_Data.csv', delimiter = ",")
Deltas = df.values[:,3:]
y_Deltas = np.ones(len(Deltas[:,0]),'i')
Deltas = np.column_stack((Deltas,y_Deltas))

ds = pd.read_csv('/home/shared/CMV/NoDelta_Data_Sample.csv', delimiter = ",")
NoDeltas = ds.values[:,3:]
y_NoDeltas = np.zeros(len(NoDeltas[:,0]),'i')
NoDeltas = np.column_stack((NoDeltas,y_NoDeltas))



print("Shuffling Data")
np.random.shuffle(Deltas)
np.random.shuffle(NoDeltas)



trainDeltas, testDeltas = Deltas[:int(len(Deltas) * .8),:], Deltas[int(len(Deltas) * .8):,:]
trainNoDeltas, testNoDeltas = NoDeltas[:int(len(NoDeltas) * .8),:], NoDeltas[int(len(NoDeltas) * .8):,:]

train_data_sm = np.concatenate((trainDeltas,trainNoDeltas), axis=0)
test_data_sm = np.concatenate((testDeltas,testNoDeltas), axis=0)

print("Duplicating Deltas")

while trainDeltas.shape[0] < trainNoDeltas.shape[0]:
    i = rd.randint(0,trainDeltas.shape[0]-1)
    trainDeltas = np.concatenate((trainDeltas, trainDeltas[i,:][np.newaxis,:]), axis=0)

while trainDeltas.shape[0] > trainNoDeltas.shape[0]:
    i = rd.randint(0,trainNoDeltas.shape[0]-1)
    trainNoDeltas = np.concatenate((trainNoDeltas, trainNoDeltas[i,:][np.newaxis,:]), axis=0)

train_data = np.concatenate((trainDeltas,trainNoDeltas), axis=0)
test_data = np.concatenate((testDeltas,testNoDeltas), axis=0)

print("# of train: ", len(train_data))
print("# of test: ", len(test_data))

print("Splitting Test Data into Deltas and No Deltas")


x_train = train_data[:,:-1]
y_train = train_data[:,-1]
y_train = y_train.astype('int')

x_train_sm = train_data_sm[:,:-1]
y_train_sm = train_data_sm[:,-1]
y_train_sm = y_train_sm.astype('int')

print("X Train, Y Train: ", x_train.shape, y_train.shape)
print("X Train SM, Y Train SM: ", x_train_sm.shape, y_train_sm.shape)


x_test = test_data[:,:-1]
y_test = test_data[:,-1]
y_test = y_test.astype('int')

# x_testDeltas = testDeltas[:,:-1]
# y_testDeltas = testDeltas[:,-1]
# y_testDeltas = y_testDeltas.astype('int')
#
# x_testNoDeltas = testNoDeltas[:,:-1]
# y_testNoDeltas = testNoDeltas[:,-1]
# y_testNoDeltas = y_testNoDeltas.astype('int')
#
#
# print("Deltas, NoDeltas = ", len(y_testDeltas), len(y_testNoDeltas))
# print("Deltas, NoDeltas = ", Deltas.shape, NoDeltas.shape)



sm = SMOTE()
x_res, y_res = sm.fit_resample(x_train_sm, y_train_sm)


print("SM Deltas, SM NoDeltas = ", len(np.where(y_res == 1)[0]), len(np.where(y_res == 0)[0]))

kwards = {'num_workers':1, 'pin_memory':True} if use_cuda else {}

tensor_data = torch.from_numpy(x_res)
tensor_data = tensor_data.float()
tensor_target = torch.from_numpy(y_res)
tensor_target = tensor_target.long()
train_data = torch.utils.data.TensorDataset(tensor_data, tensor_target)

print(x_test)
x_test = x_test.astype(np.float64)
print(x_test)
print("x_test size:", x_test.size)

tensor_data = torch.from_numpy(x_test)
tensor_data = tensor_data.float()
tensor_target = torch.from_numpy(y_test)
tensor_target = tensor_target.long()
test_data = torch.utils.data.TensorDataset(tensor_data, tensor_target)

# train_loader = torch.utils.data.DataLoader(train_loader, batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(train_loader, batch_size=args.batch_size, shuffle=True, **kwargs)






# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 1          # rnn time step / image height
INPUT_SIZE = 13         # rnn input size / image width
LR = 0.01               # learning rate
DOWNLOAD_MNIST = True   # set to True if haven't download the data


# Mnist digital dataset
# train_data = dsets.MNIST(
#     root='./mnist/',
#     train=True,                         # this is training data
#     transform=transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
#                                         # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
#     download=DOWNLOAD_MNIST,            # download it if you don't have it
# )

print(train_data)

# plot one example
# print(train_data.train_data.size()     # (60000, 28, 28)
# print(train_data.train_labels.size())   # (60000)

# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

# Data Loader for easy mini-batch return in training
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

# convert test data into Variable, pick 2000 samples to speed up testing

# test_data = dsets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())

# test_x = test_loader.test_data.type(torch.FloatTensor)[:]   # shape (2000, 28, 28) value in range(0,1)
# test_y = test_loader.test_labels.numpy()[:]    # covert to numpy array
test_x, test_y = x_test, y_test



class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=5,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(5, 1)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        print(b_x.size())      # gives batch data
        b_x = b_x.view(-1, TIME_STEP, INPUT_SIZE)           # reshape x to (batch, time_step, input_size)
        print(b_x.size())
        print(b_y.size())
        output = rnn(b_x)                               # rnn output
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        if step % 50 == 0:
            test_output = rnn(test_x)                   # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

# print 10 predictions from test data
test_output = rnn(test_x[:10].view(-1, TIME_STEP, INPUT_SIZE))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
