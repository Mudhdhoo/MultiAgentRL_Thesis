import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from NN2 import NN
import random
import itertools
from collections import deque
#from WarehouseEnv import WarehouseEnv

class Agent:
    """
    The Agent class.

    --------------- Parameters --------------- 
    init_pos:tuple
        Contains the initial starting coordinates of the agent.

    size:int
        Specifies the size of the agent, which is an n by n square.

    texture:pygame.image
        The image to be drawn on the agent.
    """
    def __init__(self, init_pos:tuple, size:int, texture:pygame.image, num_agents, gamma, batch_size, buffer_size,
                  min_replay_size, eps_max, eps_min, eps_dec, update_freq) -> None:
        self.rect = pygame.Rect(init_pos[0], init_pos[1], size, size)
        self.size = size
        self.texture = texture
        self.position = [self.rect.x, self.rect.y]
        self.has_package = False
        self.has_delivered = False
        self.state = []
        self.brain = 'super AI'
        self.replay_buffer = deque(maxlen=5000) #Replaybuffer creation
        self.rew_buffer = deque([0.0], maxlen=100)  #Rewardbuffer
        self.min_replay_size = min_replay_size  #Minimum amount of transitions in replay before training
        self.episode_reward = 0.0 #Reward in the current episode
        self.obs = 0
        self.actions = [0,1,2,3]#["Left","Up","Right","Down"]
        self.done = False
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_freq = update_freq

        self.net_online = NN(((num_agents*2)+1)) #Online network
        self.net_target = NN(((num_agents*2)+1)) #Target network

        self.optimizer = optim.Adam(self.net_online.parameters(), lr=5e-4)
        self.net_target.load_state_dict(self.net_online.state_dict()) #Set targetparameter to onlineparameters

    def init_buffer(self, warehouse): #Constructs and fills the buffer for the agent
        state = self.state  #Saves initial state
        for i in range(self.min_replay_size): #Iterate as many times as the requiered memory amount
            action = random.sample(self.actions,1) #Random action

            rew, new_state, done = warehouse.step(action, self) #Step and save ifor
            transition = (state, action, rew, done, new_state) #To tuple
            self.replay_buffer.append(transition) #Save to memory
            state = new_state #Update current state

            if done:
                state = warehouse.reset() #If done reset

    def grad_step(self,step):
        transitions = random.sample(self.replay_buffer, self.batch_size) #Sample random batch from memory
        #print(transitions)
        #for t in transitions:
        #    print(t[1])
        states = np.asarray([t[0] for t in transitions]) #States as nparray
        actions = np.asarray([t[1] for t in transitions])   #actions as nparray
        #print(actions)
        rewards = np.asarray([t[2] for t in transitions])   #rewards as nparray
        dones = np.asarray([t[3] for t in transitions]) #done as nparray
        new_states = np.asarray([t[4] for t in transitions])    #new states as nparray

        states_t = torch.as_tensor(states, dtype=torch.float32)     #Convert to torch-tensor
        actions_t = torch.as_tensor(actions, dtype=torch.int64)  #Convert to torch-tensor
        #print(actions_t)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1) #Convert to torch-tensor
        dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1) #Convert to torch-tensor
        new_states_t = torch.as_tensor(new_states, dtype=torch.float32) #Convert to torch-tensor
        
        with torch.no_grad():
            target_q_values = self.net_target(new_states_t) #Evaluate q-values for target network
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0] #Choose maximum

        targets = rewards_t + self.gamma*(1-dones_t)*max_target_q_values #Calculate target

        q_values = self.net_online(states_t) #Evaluate q-values for online network
        #print(q_values)
        action_q_values = torch.gather(input=q_values, dim=1, index = actions_t) #Gathers q-value for the action the agent took

        loss = nn.functional.mse_loss(action_q_values, targets) #Huber loss, maybe should be changed, but this seems to be used normally

        self.optimizer.zero_grad() #Zero out gradients
        loss.backward() #Calculate gradients
        self.optimizer.step #Apply them

        if step % self.update_freq == 0: #Updates target-network as frequently as specified
            self.net_target.load_state_dict(self.net_online.state_dict())

        if step % 1000 == 0: #Prints during learning
            print()
            print('Step', step)
            print('Avg reward', np.mean(self.rew_buffer))

    def move(self, button_pressed):
        if button_pressed == 0:
            self.move_left()
        elif button_pressed == 2: 
            self.move_right()
        elif button_pressed == 1:
            self.move_up()
        elif button_pressed == 3:
            self.move_down()

    def move_up(self):
        self.rect.y -= self.size
        self.position[1] = self.rect.y

    def move_down(self):
        self.rect.y += self.size
        self.position[1] = self.rect.y

    def move_left(self):
        self.rect.x -= self.size
        self.position[0] = self.rect.x

    def move_right(self):
        self.rect.x += self.size
        self.position[0] = self.rect.x

    def communicate(self, other_agents:list):
        """
        Updates the agent state by gathering information about other agents coordinates. Updates pickup status
        """
        self.state = np.array([agent.position for agent in other_agents]).flatten()     # Get every other agents position
        self.state = np.append(self.state, self.has_package)     # Get pickup status 
