from WarehouseEnv import WarehouseEnv
import pygame
from levels import *

def draw(env):
    env.WINDOW.blit(env.FLOOR_TEXTURE, (0,0))
    for wall in env.level:
        env.WINDOW.blit(wall.texture, (wall.wall_element.x, wall.wall_element.y))
    env.WINDOW.blit(env.agent.texture, (env.agent.rect.x, env.agent.rect.y))
    pygame.display.update()

def main():
    pygame.init()
    print()
    BLOCK_SIZE = 20

    FPS = 60

    warehouse = WarehouseEnv(level, BLOCK_SIZE)
    clock = pygame.time.Clock()
    playing = True

    while playing:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                playing = False

            if event.type == pygame.KEYDOWN:
                button_pressed = event.key
                warehouse.step(button_pressed)

        draw(warehouse)

    pygame.quit()

if __name__ =='__main__':
    main()