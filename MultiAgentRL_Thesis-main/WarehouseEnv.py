import pygame
from Wall import WallElement
from Agent import Agent
from colors import *
import os

class WarehouseEnv:
    def __init__(self, layout, block_size) -> None:
        self.LAYOUT = layout
        self.WIDTH = len(layout[0])*block_size
        self.HEIGHT = len(layout)*block_size
        self.BLOCK_SIZE = block_size   
        self.WALL_TEXTURE = pygame.image.load(os.path.join('images', 'wood.png'))
        self.WALL_TEXTURE = pygame.transform.scale(self.WALL_TEXTURE, (block_size, block_size))
        self.AGENT_TEXTURE = pygame.image.load(os.path.join('images', 'robot.jpg'))
        self.AGENT_TEXTURE = pygame.transform.scale(self.AGENT_TEXTURE, (block_size, block_size))
        self.FLOOR_TEXTURE = pygame.image.load(os.path.join('images', 'floor1.jpg'))
        self.FLOOR_TEXTURE = pygame.transform.scale(self.FLOOR_TEXTURE, (self.WIDTH, self.HEIGHT))
        self.PICKUP_TEXTURE = pygame.image.load(os.path.join('images', 'pickup.png'))
        self.PICKUP_TEXTURE = pygame.transform.scale(self.PICKUP_TEXTURE, (block_size, block_size))
        self.DELIVERY_TEXTURE = pygame.image.load(os.path.join('images', 'delivery.png'))
        self.DELIVERY_TEXTURE = pygame.transform.scale(self.DELIVERY_TEXTURE, (block_size, block_size))
        self.agent = Agent((block_size, block_size), block_size, self.AGENT_TEXTURE)
        self.WINDOW = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.level = []
        self.make_level()

    def make_level(self):
        x = 0
        y = 0
        for row in self.LAYOUT:
            for col in row:
                if col == "W":
                    self.level.append(WallElement((x,y), self.BLOCK_SIZE, self.WALL_TEXTURE))
                if col == "P":
                    self.level.append(WallElement((x,y), self.BLOCK_SIZE, self.PICKUP_TEXTURE))
                if col == 'D':
                    self.level.append(WallElement((x,y), self.BLOCK_SIZE, self.DELIVERY_TEXTURE))
                x += self.BLOCK_SIZE
            y += self.BLOCK_SIZE
            x = 0        

    def step(self, button_pressed):
        # Move agent
        self.agent.move(button_pressed)

        # Check for collisions
        for wall in self.level:
            if pygame.Rect.colliderect(self.agent.rect, wall.wall_element) and wall.texture == self.WALL_TEXTURE:
                if button_pressed == pygame.K_LEFT:
                    self.agent.move_right()
                elif button_pressed == pygame.K_RIGHT: 
                    self.agent.move_left()
                elif button_pressed == pygame.K_UP:
                    self.agent.move_down()
                elif button_pressed == pygame.K_DOWN:
                    self.agent.move_up()
