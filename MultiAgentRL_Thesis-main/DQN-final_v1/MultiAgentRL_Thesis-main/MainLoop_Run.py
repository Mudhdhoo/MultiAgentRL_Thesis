from WarehouseEnv import WarehouseEnv
import pygame
from levels import *
import itertools
import random
import numpy as np

def main():
    #Parameters doesnt do anything but are required to use the environment
    gamma=0.99 #Discountfactor
    batch_size = 50 #Amount of sampled transitions from replaybuffer
    buffer_size = 10000, #Max number of transitions stored
    min_replay_size = 1000  #Minimum amount of transitions in replay before training
    eps = 0 #Want deterministic while running with trained networks
    eps_min = 0.1
    eps_dec = 500000
    update_freq = 500 #How often targetparameters are copied fron online
    pygame.init()

    BLOCK_SIZE = 20
    NUM_AGENTS = 1
    FPS = 5

    warehouse = WarehouseEnv(test_lvl3, BLOCK_SIZE, NUM_AGENTS, gamma, batch_size, buffer_size,
                  min_replay_size, eps, eps_min, eps_dec, update_freq, "test_lvl3") #Environment initialized
    clock = pygame.time.Clock()
    playing = True

    for agent in warehouse.agents:  #Load the trained networks for current settings
        agent.load_model()

    warehouse.reset() #Reset the warehouse before starting learning
    episode = 1
    while playing:
        for event in pygame.event.get():
            for step in itertools.count(): #Counts how many states has been visited
                clock.tick(FPS)
                for agent in warehouse.agents: #For each agent, perform actions
                    if agent.done == False: #If agent done, stop making decisions
                        action = agent.net_online.act(agent.state)#Deterministic acting after training
                        rew, new_state, agent.done = warehouse.step(action[0], agent) #Take step, save reward, new state and if done

                        agent.episode_reward += rew #Save reward for current episode
                done_check = []
                for agent in warehouse.agents: #Checks if any agent still working
                    done_check.append(agent.done)
                if False not in done_check: #If not, reset environment and save episodereward to memory
                    episode += 1
                    warehouse.reset()
                    print("Episode "+str(episode))
                    for agent in warehouse.agents:
                        print("Agent "+str(agent.index))
                        print("Successful Deliveries: " + str(agent.delivered_packages), "Crashes: " + str(agent.crashes))
                        agent.rew_buffer.append(agent.episode_reward)
                        agent.episode_reward = 0.0
                warehouse.render()
    pygame.quit()
            

if __name__ =='__main__':
    main()