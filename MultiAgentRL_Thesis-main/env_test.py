from WarehouseEnv import WarehouseEnv
import pygame
from levels import *

def main():
    pygame.init()

    BLOCK_SIZE = 20
    NUM_AGENTS = 5
    FPS = 60

    warehouse = WarehouseEnv(level, BLOCK_SIZE, NUM_AGENTS)
    clock = pygame.time.Clock()
    playing = True

    while playing:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                playing = False

            if event.type == pygame.KEYDOWN:
                button_pressed = event.key
                agent = warehouse.agents[0]
                reward, next_state = warehouse.step(button_pressed, agent)
                print(reward, next_state)

        warehouse.render()

    pygame.quit()

if __name__ =='__main__':
    main()