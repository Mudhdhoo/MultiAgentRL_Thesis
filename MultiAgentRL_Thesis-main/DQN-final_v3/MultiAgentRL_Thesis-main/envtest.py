from WarehouseEnv import WarehouseEnv
import pygame
from levels import *

def main():
    pygame.init()

    BLOCK_SIZE = 20
    NUM_AGENTS = 1
    FPS = 60
    gamma=0.95 #Discountfactor
    batch_size = 50 #Amount of sampled transitions from replaybuffer
    buffer_size = 15000 #Max number of transitions stored
    min_replay_size = 7500  #Minimum amount of transitions in replay before training
    eps_max = 1
    eps = 1.0 #Specifictations of epsilon
    eps_min = 0.1
    eps_dec = 1000000
    update_freq = 150 #How often targetparameters are copied fron online

    warehouse = WarehouseEnv(test_lvl3, BLOCK_SIZE, NUM_AGENTS, gamma, batch_size, buffer_size,
                  min_replay_size, eps, eps_min, eps_dec, update_freq, "test_lvl3")
    clock = pygame.time.Clock()
    playing = True

    while playing:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                playing = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    button_pressed = 2
                if event.key == pygame.K_LEFT:
                    button_pressed = 0
                if event.key == pygame.K_DOWN:
                    button_pressed = 3
                if event.key == pygame.K_UP:
                    button_pressed = 1
                agent = warehouse.agents[0]
                reward, next_state, done = warehouse.step(button_pressed, agent)
                if done:
                    warehouse.reset()
                print(reward, agent.state, next_state, len(next_state))
            
                       # Reset the environment if an agent crashes into a wall or some other agent


        warehouse.render()

    pygame.quit()

if __name__ =='__main__':
    main()