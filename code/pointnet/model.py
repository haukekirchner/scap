"""Implementation of the PointNet (Qi, 2017). Code is adapted from Nikita Karaev
https://towardsdatascience.com/deep-learning-on-point-clouds-implementing-pointnet-in-google-colab-1fd65cd3a263."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNet(nn.Module):
    """PointNet (Qui et al. 2017) implementation for classification."""

    def __init__(self, classes):
        super().__init__()
        self.transform = self.Transform(self.Tnet(k=3), self.Tnet(k=64))
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    class Tnet(nn.Module):
        def __init__(self, k=3):
            super().__init__()
            self.k = k
            self.conv1 = nn.Conv1d(k, 64, 1)
            self.conv2 = nn.Conv1d(64, 128, 1)
            self.conv3 = nn.Conv1d(128, 1024, 1)
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, k*k)

            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)

        def forward(self, input):
            bs = input.size(0)
            xb = F.relu(self.bn1(self.conv1(input)))
            xb = F.relu(self.bn2(self.conv2(xb)))
            xb = F.relu(self.bn3(self.conv3(xb)))
            pool = nn.MaxPool1d(xb.size(-1))(xb)
            flat = nn.Flatten(1)(pool)
            xb = F.relu(self.bn4(self.fc1(flat)))
            xb = F.relu(self.bn5(self.fc2(xb)))

            # initialize as identity
            init = torch.eye(self.k, requires_grad=True).repeat(bs, 1, 1)
            if xb.is_cuda:
                init = init.cuda()
            matrix = self.fc3(xb).view(-1, self.k, self.k) + init
            return matrix

    class Transform(nn.Module):
        def __init__(self, input_transform, feature_transform):
            super().__init__()
            self.input_transform = input_transform
            self.feature_transform = feature_transform
            self.conv1 = nn.Conv1d(3, 64, 1)

            self.conv2 = nn.Conv1d(64, 128, 1)
            self.conv3 = nn.Conv1d(128, 1024, 1)

            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)

        def forward(self, input):
            matrix3x3 = self.input_transform(input)
            # batch matrix multiplication
            xb = torch.matmul(torch.transpose(input, 1, 2),
                           matrix3x3).transpose(1, 2)

            xb = F.relu(self.bn1(self.conv1(xb)))

            matrix64x64 = self.feature_transform(xb)
            xb = torch.matmul(torch.transpose(xb, 1, 2),
                           matrix64x64).transpose(1, 2)

            xb = F.relu(self.bn2(self.conv2(xb)))
            xb = self.bn3(self.conv3(xb))
            xb = nn.MaxPool1d(xb.size(-1))(xb)
            output = nn.Flatten(1)(xb)
            return output, matrix3x3, matrix64x64

    def forward(self, input):
        input = input.transpose(1, 2).to(dtype=torch.float)
        xb, matrix3x3, matrix64x64 = self.transform(input)
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        return self.logsoftmax(output), matrix3x3, matrix64x64


def pointnetloss(outputs, labels, m3x3, m64x64, alpha=0.001, device=None):
    """Loss containing the classification loss (negative log likelihood) and the feature transformation (regularization term to orthogonal matrix)."""
    criterion = torch.nn.NLLLoss()
    bs = outputs.size(0)
    # Calculate difference to identity matrix for regularization.
    id3x3 = torch.eye(3, requires_grad=True, device=device).repeat(bs, 1, 1)
    id64x64 = torch.eye(64, requires_grad=True, device=device).repeat(bs, 1, 1)
    diff3x3 = id3x3 - torch.matmul(m3x3, m3x3.transpose(1, 2))
    diff64x64 = id64x64 - torch.matmul(m64x64, m64x64.transpose(1, 2))
    # Negative log likelihood criterion is already adapted to batch size.
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3) + torch.norm(diff64x64)) / float(bs)


def fix_model_params(model_params, dictionary_params):
    """Checks whether prefix.sth in dictionary_params == sth in model_params."""
    correct_params = {}
    for (key_model, _), (key_dict, val_dict) in zip(model_params.state_dict().items(),
                                                    dictionary_params.items()):

        if key_dict == key_model:
            correct_params[key_model] = val_dict

        # Assert that the right parameters are assigned to the correct layer.
        # All of the parameters in the dictionary contain an unnecessary .module.
        elif key_dict.split('.', maxsplit=1)[1] == key_model:
            correct_params[key_model] = val_dict

        else:
            raise ValueError(
                f"Cannot assign parameters of {key_dict} to {key_model}!")

    # All dictionary parameters should be corrected.
    assert len(correct_params) == len(dictionary_params)
    return correct_params
