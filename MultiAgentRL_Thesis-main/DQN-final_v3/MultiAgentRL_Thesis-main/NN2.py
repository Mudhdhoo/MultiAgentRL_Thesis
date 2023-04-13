import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

class NN(nn.Module):
    
    def __init__(self, state_in_length, q_out = 4):
        super(NN, self).__init__()
        self.layer1 = nn.Linear(state_in_length, 64)
        self.layer2 = nn.Linear(64,64)
        self.layer3 = nn.Linear(64,q_out)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu')
        self.to(self.device)
        
    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
    def act(self, state):
        state_t = torch.as_tensor(state, device=self.device, dtype = torch.float32)
        q_values = self(state_t.unsqueeze(0))

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = [max_q_index.detach().item()]

        return action
