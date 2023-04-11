from WarehouseEnv import WarehouseEnv
import pygame
from levels import *
import itertools
import random
import numpy as np

def main():
    gamma=0.95 #Discountfactor
    batch_size = 50 #Amount of sampled transitions from replaybuffer
    buffer_size = 10000 #Max number of transitions stored
    min_replay_size = 5000  #Minimum amount of transitions in replay before training
    eps_max = 1
    eps = 1.0 #Specifictations of epsilon
    eps_min = 0.1
    eps_dec = 1000000
    update_freq = 100 #How often targetparameters are copied fron online
    pygame.init()

    BLOCK_SIZE = 20
    NUM_AGENTS = 1
    FPS = 10

    warehouse = WarehouseEnv(test_lvl3, BLOCK_SIZE, NUM_AGENTS, gamma, batch_size, buffer_size,
                  min_replay_size, eps, eps_min, eps_dec, update_freq, "test_lvl3") #Environment initialized
    playing = True
    for agent in warehouse.agents: #Initialize each agents memory
        agent.init_buffer(warehouse)

    warehouse.reset() #Reset the warehouse before starting learning
    episode_max = 50
    episode_start = 0
    episode = 1
    while playing:
        
        for event in pygame.event.get():
            for step in itertools.count(): #Counts how many states has been visited
                #eps = np.interp(step, [0, eps_dec], [eps_max, eps_min]) #For each step, decay epsilon
                if eps > eps_min:
                    eps = np.exp(-step/(eps_dec/2.5))
                #eps = 0.999999**step
                for agent in warehouse.agents: #For each agent, perform actions
                    if agent.done == False: #If agent done, stop making decisions
                        state = agent.state #Save state
                        random_sample = random.random()
                        if random_sample <= eps: #Choose if eploit or explore
                            action = random.sample(agent.actions, 1)
                        else:
                            action = agent.net_online.act(agent.state)
                        rew, new_state, agent.done = warehouse.step(action[0], agent) #Take step, save reward, new state and if done
                        transition = (state, action, rew, agent.done, new_state) #Save in tuple
                        agent.replay_buffer.append(transition) #Save to memory

                        agent.episode_reward += rew #Save reward for current episode
                done_check = []
                for agent in warehouse.agents: #Checks if any agent still working
                    done_check.append(agent.done)
                if False not in done_check or step >= (episode_start+episode_max): #If not, reset environment and save episodereward to memory
                    episode += 1
                    warehouse.reset()
                    episode_start = step
                    for agent in warehouse.agents:
                        agent.rew_buffer.append(agent.episode_reward)
                        agent.episode_reward = 0.0
                if step<=eps_dec:
                    for agent in warehouse.agents: #When all agents has taken a step, train their networks
                        agent.grad_step(step)
                if step == eps_dec:
                    for agent in warehouse.agents:
                        agent.plot_reward()
                        agent.save_model()
                        agent.crashes = 0
                        agent.delivered_packages = 0
                    print(episode)
                    return
    pygame.quit()
            

if __name__ =='__main__':
    main()