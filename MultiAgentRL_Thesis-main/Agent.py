import pygame
import numpy as np

class Agent:
    def __init__(self, init_pos:tuple, size:int, texture) -> None:
        self.rect = pygame.Rect(init_pos[0], init_pos[1], size, size)
        self.size = size
        self.texture = texture
        self.position = [self.rect.x, self.rect.y]
        self.has_package = False
        self.state = []
        self.brain = 'super AI'

    def move(self, button_pressed):
        if button_pressed == pygame.K_LEFT:
            self.move_left()
        elif button_pressed == pygame.K_RIGHT: 
            self.move_right()
        elif button_pressed == pygame.K_UP:
            self.move_up()
        elif button_pressed == pygame.K_DOWN:
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

    def communicate(self, other_agents):
        self.state = np.array([agent.position for agent in other_agents]).flatten()     # Get every other agents position
        self.state = np.append(self.state, self.has_package)     # Get pickup status 
