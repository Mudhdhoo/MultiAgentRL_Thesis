import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from NN2 import NN
import random
import itertools
from collections import deque
import matplotlib.pyplot as plt
import csv
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
                  min_replay_size, eps_max, eps_min, eps_dec, update_freq, index, level_name) -> None:
        self.rect = pygame.Rect(init_pos[0], init_pos[1], size, size)
        self.starting_point = (init_pos[0],init_pos[1])
        self.size = size
        self.texture = texture
        self.position = [self.rect.x, self.rect.y]
        self.has_package = False
        self.has_delivered = False
        self.state = []
        self.reward = 0
        self.action = None
        self.old_state = None
        self.brain = 'super AI'
        self.replay_buffer = deque([],maxlen=int(buffer_size)) #Replaybuffer creation
        self.rew_buffer = deque([0.0], maxlen=100)  #Rewardbuffer
        self.min_replay_size = min_replay_size  #Minimum amount of transitions in replay before training
        self.episode_reward = 0.0 #Reward in the current episode
        self.obs = 0
        self.actions = [0,1,2,3]#["Left","Up","Right","Down"]
        self.done = False
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_freq = update_freq
        self.index = index
        self.num_agents = num_agents
        self.level_name = level_name
        self.buffer_size = buffer_size
        self.lr = 1e-3
        self.crash = False
        self.done_saved = False

        self.net_online = NN(((num_agents*2)+1)) #Online network
        self.net_target = NN(((num_agents*2)+1)) #Target network

        self.optimizer = optim.Adam(self.net_online.parameters(), lr=self.lr)
        self.net_target.load_state_dict(self.net_online.state_dict()) #Set targetparameter to onlineparameters

        self.delivered_packages = 0
        self.crashes = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu')

        self.eps_dec = eps_dec
        self.rew_for_plot = []
        self.loss_for_plot = []
        self.step_for_plot = []


    #def init_buffer(self, warehouse): #Constructs and fills the buffer for the agent
    #    #state = self.state  #Saves initial state
    #    for i in range(self.min_replay_size): #Iterate as many times as the requiered memory amount
    #        self.action = random.sample(self.actions,1)[0] #Random action
    #        self.old_state = self.state
    #        warehouse.step() #Step and save ifor
    #        transition = (self.old_state, self.action, self.reward, self.done, self.state) #To tuple
    #        self.replay_buffer.append(transition) #Save to memory
    #        #state = new_state #Update current state
#
    #        if self.done:
    #            warehouse.reset() #If done reset

    def grad_step(self,step):
        transitions = random.sample(self.replay_buffer, self.batch_size) #Sample random batch from memory
        states = np.asarray([t[0] for t in transitions]) #States as nparray
        actions = np.asarray([t[1] for t in transitions])   #actions as nparray
        rewards = np.asarray([t[2] for t in transitions])   #rewards as nparray
        dones = np.asarray([t[3] for t in transitions]) #done as nparray
        new_states = np.asarray([t[4] for t in transitions])    #new states as nparray

        states_t = torch.as_tensor(states, device=self.device, dtype=torch.float32)     #Convert to torch-tensor
        actions_t = torch.as_tensor(actions, device=self.device, dtype=torch.int64).unsqueeze(-1)  #Convert to torch-tensor
        rewards_t = torch.as_tensor(rewards, device=self.device, dtype=torch.float32).unsqueeze(-1) #Convert to torch-tensor
        dones_t = torch.as_tensor(dones, device=self.device, dtype=torch.float32).unsqueeze(-1) #Convert to torch-tensor
        new_states_t = torch.as_tensor(new_states, device=self.device, dtype=torch.float32) #Convert to torch-tensor
        
        with torch.no_grad():
            target_q_values = self.net_target(new_states_t) #Evaluate q-values for target network
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0] #Choose maximum

        targets = rewards_t + self.gamma*(1-dones_t)*max_target_q_values #Calculate target

        q_values = self.net_online(states_t) #Evaluate q-values for online network
        action_q_values = torch.gather(input=q_values, dim=1, index = actions_t) #Gathers q-value for the action the agent took

        loss = nn.functional.smooth_l1_loss(action_q_values, targets) #Huber loss, maybe should be changed, but this seems to be used normally

        self.optimizer.zero_grad() #Zero out gradients
        loss.backward() #Calculate gradients
        self.optimizer.step() #Apply them
        

        if step % self.update_freq == 0: #Updates target-network as frequently as specified
            self.net_target.load_state_dict(self.net_online.state_dict())
            self.loss_for_plot.append(float(loss))
            self.rew_for_plot.append(np.mean(self.rew_buffer))
            self.step_for_plot.append(step)
            #print(loss) #Uncomment if loss function is preffered to be be printed

        if step % 10000 == 0: #Prints during learning
            print()
            print('Step', step)
            print('Avg reward', np.mean(self.rew_buffer))
            
    def plot(self):
            #Plots average reward against step
            plt.plot(self.step_for_plot,self.rew_for_plot, linestyle = 'dashed', linewidth = 2, marker = 'o', markerfacecolor = 'blue', markersize =5)
            plt.ylim(min(self.rew_for_plot),0)
            plt.xlim(0,self.step_for_plot[-1])

            plt.xlabel('Step')
            plt.ylabel('Avg Reward')
            plt.title('Average reward against steps')

            plt.savefig(F'AvgReward_{self.level_name}_{self.eps_dec}Steps_{self.update_freq}_Batchsize{self.batch_size}_lr{self.lr}_Num_Agents_{self.num_agents}.png')

            plt.plot(self.step_for_plot,self.loss_for_plot, linestyle = '-', linewidth = 1)
            plt.ylim(0,max(self.loss_for_plot))
            plt.xlim(0,self.step_for_plot[-1])

            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title('Loss against steps')
            plt.savefig(F'Loss_{self.level_name}_{self.eps_dec}Steps_{self.update_freq}_Batchsize{self.batch_size}_lr{self.lr}_Num_Agents_{self.num_agents}.png')

            with open(F"Level_{self.level_name}_Steps_{self.eps_dec}_Buffersize_{self.buffer_size}_Batchsize_{self.batch_size}_Lr_{self.lr}_Num_Agents_{self.num_agents}_Trained_Network_Agent{self.index}.csv", 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Step", "Avg Reward", "Loss"])
                for i in range(len(self.step_for_plot)):
                    writer.writerow([self.step_for_plot[i], self.rew_for_plot[i], self.loss_for_plot[i]])

    def save_model(self):
        #Saves the model
        torch.save(self.net_online.state_dict(), F"Level_{self.level_name}_Steps_{self.eps_dec}_Buffersize_{self.buffer_size}_Batchsize_{self.batch_size}_Lr_{self.lr}_Num_Agents_{self.num_agents}_Trained_Network_Agent{self.index}")

    def load_model(self):
        #Loads trained models
        model = NN(((self.num_agents*2)+1))
        model.load_state_dict(torch.load(F"Level_{self.level_name}_Steps_{self.eps_dec}_Buffersize_{self.buffer_size}_Batchsize_{self.batch_size}_Lr_{self.lr}_Num_Agents_{self.num_agents}_Trained_Network_Agent{self.index}",map_location=self.device))
        self.net_online = model

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
        self.state = np.array([[int(agent.position[0]/20),int(agent.position[1]/20)] for agent in other_agents]).flatten()     # Get every other agents position
        self.state = np.append(self.state, self.has_package)     # Get pickup status 
