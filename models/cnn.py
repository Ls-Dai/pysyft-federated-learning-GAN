import torch.nn as nn
import torch.nn.functional as func

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = func.relu(self.fc1(x))
        x = self.fc2(x)
        return func.log_softmax(x, dim=1)