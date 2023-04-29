from WarehouseEnv import WarehouseEnv
import pygame
from levels import *
import itertools
import random
import numpy as np

def main():
    #Parameters doesnt do anything but are required to use the environment
    gamma=0.95 #Discountfactor
    batch_size = 50 #Amount of sampled transitions from replaybuffer
    buffer_size = 50000 #Max number of transitions stored
    min_replay_size = 25000  #Minimum amount of transitions in replay before training
    eps = 0 #Want deterministic while running with trained networks
    eps_min = 0.1
    eps_dec = 200000
    update_freq = 150 #How often targetparameters are copied fron online
    pygame.init()

    BLOCK_SIZE = 20
    NUM_AGENTS = 3
    FPS = 150

    warehouse = WarehouseEnv(test_lvl34_2, BLOCK_SIZE, NUM_AGENTS, gamma, batch_size, buffer_size,
                  min_replay_size, eps, eps_min, eps_dec, update_freq, "test_lvl34_2") #Environment initialized
    clock = pygame.time.Clock()
    playing = True

    for agent in warehouse.agents:  #Load the trained networks for current settings
        agent.load_model()

    warehouse.reset() #Reset the warehouse before starting learning
    episode = 1
    successful_episodes = 0
    crashed_episodes = 0
    while playing:
        for event in pygame.event.get():
            for step in itertools.count(): #Counts how many states has been visited
                clock.tick(FPS)
                for agent in warehouse.agents: #For each agent, perform actions
                    if agent.done == False: #If agent done, stop making decisions
                        agent.action = agent.net_online.act(agent.state)[0]#Deterministic acting after training
                warehouse.step() #Take step, save reward, new state and if done
                for agent in warehouse.agents:
                    agent.episode_reward += agent.reward #Save reward for current episode
                done_check = []
                crash_check = []
                for agent in warehouse.agents: #Checks if any agent still working
                    done_check.append(agent.done)
                    crash_check.append(agent.crash)
                if False not in done_check or True in crash_check: #If not, reset environment and save episodereward to memory
                    if False not in done_check:
                        successful_episodes += 1
                    if True in crash_check:
                        crashed_episodes += 1
                    warehouse.reset()
                    print("Episode "+str(episode))
                    print("Successful Episodes: "+str(successful_episodes)+" Crashed Episodes: "+str(crashed_episodes))
                    episode += 1
                    for agent in warehouse.agents:
                        agent.rew_buffer.append(agent.episode_reward)
                        agent.episode_reward = 0.0
                warehouse.render()
                if episode > 500:
                    return
    pygame.quit()
            

if __name__ =='__main__':
    main()