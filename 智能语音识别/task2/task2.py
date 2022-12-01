import os
from socket import AddressFamily
import wave
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from python_speech_features import mfcc
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
#from sklearn_features.transformers import DataFrameSelector
from torch import Tensor
from torch.optim import SGD #Adam
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import Linear, ReLU, Sigmoid, Module, BCELoss
from torch.nn.init import kaiming_uniform_, xavier_uniform_
from torch import nn
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F
def parse_vad_label(line, frame_size: float = 0.032, frame_shift: float = 0.008):
    """Parse VAD information in each line, and convert it to frame-wise VAD label.

    Args:
        line (str): e.g. "0.2,3.11 3.48,10.51 10.52,11.02"
        frame_size (float): frame size (in seconds) that is used when
                            extarcting spectral features
        frame_shift (float): frame shift / hop length (in seconds) that
                            is used when extarcting spectral features
    Returns:
        frames (List[int]): frame-wise VAD label

    Examples:
        >>> label = parse_vad_label("0.3,0.5 0.7,0.9")
        [0, ..., 0, 1, ..., 1, 0, ..., 0, 1, ..., 1]
        >>> print(len(label))
        110

    NOTE: The output label length may vary according to the last timestamp in `line`,
    which may not correspond to the real duration of that sample.

    For example, if an audio sample contains 1-sec silence at the end, the resulting
    VAD label will be approximately 1-sec shorter than the sample duration.

    Thus, you need to pad zeros manually to the end of each label to match the number
    of frames in the feature. E.g.:
        >>> feature = extract_feature(audio)    # frames: 320
        >>> frames = feature.shape[1]           # here assumes the frame dimention is 1
        >>> label = parse_vad_label(vad_line)   # length: 210
        >>> import numpy as np
        >>> label_pad = np.pad(label, (0, np.maximum(frames - len(label), 0)))[:frames]
    """
    frame2time = lambda n: n * frame_shift + frame_size / 2
    frames = []
    frame_n = 0
    for time_pairs in line.split():
        start, end = map(float, time_pairs.split(","))
        assert end > start, (start, end)
        while frame2time(frame_n) < start:
            frames.append(0)
            frame_n += 1
        while frame2time(frame_n) <= end:
            frames.append(1)
            frame_n += 1
    return frames


def prediction_to_vad_label(
    prediction,
    frame_size: float = 0.032,
    frame_shift: float = 0.008,
    threshold: float = 0.5,
):
    """Convert model prediction to VAD labels.

    Args:
        prediction (List[float]): predicted speech activity of each **frame** in one sample
            e.g. [0.01, 0.03, 0.48, 0.66, 0.89, 0.87, ..., 0.72, 0.55, 0.20, 0.18, 0.07]
        frame_size (float): frame size (in seconds) that is used when
                            extarcting spectral features
        frame_shift (float): frame shift / hop length (in seconds) that
                            is used when extarcting spectral features
        threshold (float): prediction values that are higher than `threshold` are set to 1,
                            and those lower than or equal to `threshold` are set to 0
    Returns:
        vad_label (str): converted VAD label
            e.g. "0.31,2.56 2.6,3.89 4.62,7.99 8.85,11.06"

    NOTE: Each frame is converted to the timestamp according to its center time point.
    Thus the converted labels may not exactly coincide with the original VAD label, depending
    on the specified `frame_size` and `frame_shift`.
    See the following exmaple for more detailed explanation.

    Examples:
        >>> label = parse_vad_label("0.31,0.52 0.75,0.92")
        >>> prediction_to_vad_label(label)
        '0.31,0.53 0.75,0.92'
    """
    frame2time = lambda n: n * frame_shift + frame_size / 2
    speech_frames = []
    prev_state = False
    start, end = 0, 0
    end_prediction = len(prediction) - 1
    for i, pred in enumerate(prediction):
        state = pred > threshold
        if not prev_state and state:
            # 0 -> 1
            start = i
        elif not state and prev_state:
            # 1 -> 0
            end = i
            speech_frames.append(
                "{:.2f},{:.2f}".format(frame2time(start), frame2time(end))
            )
        elif i == end_prediction and state:
            # 1 -> 1 (end)
            end = i
            speech_frames.append(
                "{:.2f},{:.2f}".format(frame2time(start), frame2time(end))
            )
        prev_state = state
    return " ".join(speech_frames)


##############################################
# Examples of how to use the above functions #
##############################################
def read_label_from_file(
    path="vad/data/train_label.txt", frame_size: float = 0.032, frame_shift: float = 0.008
):
    """Read VAD information of all samples, and convert into
    frame-wise labels (not padded yet).

    Args:
        path (str): Path to the VAD label file.
        frame_size (float): frame size (in seconds) that is used when
                            extarcting spectral features
        frame_shift (float): frame shift / hop length (in seconds) that
                            is used when extarcting spectral features
    Returns:
        data (dict): Dictionary storing the frame-wise VAD
                    information of each sample.
            e.g. {
                "1031-133220-0062": [0, 0, 0, 0, ... ],
                "1031-133220-0091": [0, 0, 0, 0, ... ],
                ...
            }
    """
    data = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = line.strip().split(maxsplit=1)
            if len(sps) == 1:
                print(f'Error happened with path="{path}", id="{sps[0]}", value=""')
            else:
                k, v = sps
            if k in data:
                raise RuntimeError(f"{k} is duplicated ({path}:{linenum})")
            data[k] = parse_vad_label(v, frame_size=frame_size, frame_shift=frame_shift)
    return data


from sklearn import metrics


def compute_eer(target_scores, nontarget_scores):
    """Calculate EER following the same way as in Kaldi.

    Args:
        target_scores (array-like): sequence of scores where the
                                    label is the target class
        nontarget_scores (array-like): sequence of scores where the
                                    label is the non-target class
    Returns:
        eer (float): equal error rate
        threshold (float): the value where the target error rate
                           (the proportion of target_scores below
                           threshold) is equal to the non-target
                           error rate (the proportion of nontarget_scores
                           above threshold)
    """
    assert len(target_scores) != 0 and len(nontarget_scores) != 0
    tgt_scores = sorted(target_scores)
    nontgt_scores = sorted(nontarget_scores)

    target_size = float(len(tgt_scores))
    nontarget_size = len(nontgt_scores)
    target_position = 0
    for target_position, tgt_score in enumerate(tgt_scores[:-1]):
        nontarget_n = nontarget_size * target_position / target_size
        nontarget_position = int(nontarget_size - 1 - nontarget_n)
        if nontarget_position < 0:
            nontarget_position = 0
        if nontgt_scores[nontarget_position] < tgt_score:
            break
    threshold = tgt_scores[target_position]
    eer = target_position / target_size
    return eer, threshold


def get_metrics(prediction, label):
    """Calculate several metrics for a binary classification task.

    Args:
        prediction (array-like): sequence of probabilities
            e.g. [0.1, 0.4, 0.35, 0.8]
        labels (array-like): sequence of class labels (0 or 1)
            e.g. [0, 0, 1, 1]
    Returns:
        auc: area-under-curve
        eer: equal error rate
    """  # noqa: H405, E261
    assert len(prediction) == len(label), (len(prediction), len(label))
    fpr, tpr, thresholds = metrics.roc_curve(label, prediction, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # from scipy.optimize import brentq
    # from scipy.interpolate import interp1d
    # fnr = 1 - tpr
    # eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    eer, thres = compute_eer(
        [pred for i, pred in enumerate(prediction) if label[i] == 1],
        [pred for i, pred in enumerate(prediction) if label[i] == 0],
    )
    return auc, eer


if __name__ == "__main__":
    # 第一个参数为模型预测输出（可以是概率，也可以是二值分类结果）
    # 第二个参数为数据对应的标签
    print(get_metrics([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]))
    # 注意：计算最终指标时，应将整个数据集（而不是在每个样本上单独计算）的所有语音帧预测结果合并在一个list中，
    # 对应的标签也合并在一个list中，然后再调用get_metrics来计算指标

class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df=read_csv(path,header=None)
        # store the inputs and outputs
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        # ensure input data is floats
        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.3):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])


class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=1):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 8)
        self.bn1 = nn.BatchNorm1d(8)
        self.fc1_drop = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(8, 4)
        self.bn2 = nn.BatchNorm1d(4)
        self.fc2_drop = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(4, 2)
        self.bn3 = nn.BatchNorm1d(2)
        self.fc3_drop = nn.Dropout(p=0.2)
        
        self.last = nn.Linear(2, num_classes)
    
    def forward(self, x):
        out = F.relu(self.bn1((self.fc1(x))))
        out = F.relu(self.bn2((self.fc2(out))))
        out = F.relu(self.bn3((self.fc3(out))))
        
        out = self.last(out)
        
        return out

# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs,8)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
       # second hidden layer
        self.hidden2 = Linear(8, 4)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        
        # third hidden layer and output
        self.hidden3 = Linear(4, 1)
        xavier_uniform_(self.hidden3.weight)
        self.act3 =Sigmoid()

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        return X


# prepare the dataset
def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=256, shuffle=True)
    test_dl = DataLoader(test, batch_size=128, shuffle=False)
    return train_dl, test_dl



# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = BCELoss()       #torch.nn.CrossEntropy()
    lr=0.005
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    # enumerate epochs
    for epoch in range(6):
        # enumerate mini batches
        lr=lr/2
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            
            
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            #if epoch<1:
                #print("epoch: {}, batch: {}, loss: {}".format(epoch, i, loss.data))
            # update model weights
            optimizer.step()
        print("epoch: {}, batch: {}, loss: {}".format(epoch, i, loss.data))

# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = [], []
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc


# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat

#path = r'C:\Users\xzs\Desktop\hellokitty.csv'
path = "task2.csv"
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
model = MLP(13)
print(model)
# train the model
train_model(train_dl, model)
# evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)



#
# 
# test




csv_file_test ='dev_test.csv'
csv_data_test = pd.read_csv(csv_file_test, low_memory = False)#防止弹出警告
csv_df_test = pd.DataFrame(csv_data_test)
x_test=torch.tensor(csv_df_test.values[:,:-1]).float()
y_test=model(x_test).view(-1).tolist()
test_label=torch.tensor(csv_df_test.values[:,-1]).float().tolist()
print(get_metrics(y_test,test_label))

