import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim=3):        
        super(MLP, self).__init__()
        
        # self.mlp = nn.Sequential(
        #     nn.Linear(input_dim, 128),
        #     nn.GroupNorm(num_groups=4, num_channels=128), 
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.GroupNorm(num_groups=4, num_channels=128), 
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.GroupNorm(num_groups=4, num_channels=128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.GroupNorm(num_groups=4, num_channels=128),
        #     nn.ReLU(),
        #     nn.Linear(128, 3)
        # )
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, input_dim)
        self.relu = nn.ReLU()
        self.gnorm = nn.GroupNorm(num_groups=4, num_channels=128)
            
    def forward(self, x):
        '''
        Input: Bx3
        Output: Bx3
        '''
        o =self.relu(self.gnorm(self.fc1(x)))
        o =self.relu(self.gnorm(self.fc2(o)))
        o =self.relu(self.gnorm(self.fc3(o)))
        o =self.relu(self.gnorm(self.fc4(o)))
        return self.fc5(o) + x
        # return self.mlp(x) + x