import pygame
import random
from Wall import WallElement
from Agent import Agent
from colors import *
import os

class WarehouseEnv:
    """
    The Warehouse environment.

    --------------- Parameters --------------- 
    layout: list
        A matrix represented by lists which specifies the level layout. The environment will then 
        be a visual representation of that design.

    block_size: int
        The size each block in the environment. Note that bigger blocks do not mean a bigger environment,
        just a bigger window when rendering.

    num_agents: int
        The number of agents present in the environment.
    """

    def __init__(self, layout:list, block_size:int, num_agents:int) -> None:
        self.LAYOUT = layout
        self.WIDTH = len(layout[0])*block_size
        self.HEIGHT = len(layout)*block_size
        self.BLOCK_SIZE = block_size   
        self.NUM_AGENTS = num_agents
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
        self.WINDOW = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.STEP_REWARD = -1
        self.CRASH_REWARD = -30
        self.PICKUP_REWARD = 0
        self.DELIVERY_REWARD = 1
        self.level = []
        self.make_level()
        self.agents = [Agent(self.generate_start(), block_size, self.AGENT_TEXTURE) for _ in range(1, num_agents + 1)]
        for agent in self.agents:
            agent.communicate(self.agents)      # Initialize communication between agents

    def make_level(self) -> None:
        """
        Creates the level by building walls according to the given level layout.
        """
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

    def step(self, button_pressed, agent:Agent) -> tuple:
        """
        Steps the environment forward one time-step according to the action taken by a specified
        agent. Returns the reward received by taking that action and the new state.
        """
        reward = self.STEP_REWARD

        # Move agent
        agent.move(button_pressed)

        # Check for collisions
        for wall in self.level:
            if pygame.Rect.colliderect(agent.rect, wall.wall_element) and wall.texture == self.WALL_TEXTURE:
                reward = self.CRASH_REWARD
                if button_pressed == pygame.K_LEFT:
                    agent.move_right()
                elif button_pressed == pygame.K_RIGHT: 
                    agent.move_left()
                elif button_pressed == pygame.K_UP:
                    agent.move_down()
                elif button_pressed == pygame.K_DOWN:
                    agent.move_up()
                return reward, agent.state

            if pygame.Rect.colliderect(agent.rect, wall.wall_element) and wall.texture == self.PICKUP_TEXTURE and not agent.has_package:
                reward = self.PICKUP_REWARD
                agent.has_package = True

            if pygame.Rect.colliderect(agent.rect, wall.wall_element) and wall.texture == self.DELIVERY_TEXTURE and agent.has_package:
                reward = self.DELIVERY_REWARD
                agent.has_package = False

        agent.communicate(self.agents)      # Get state information from other agents

        return reward, agent.state

    def reset(self) -> None:
        """
        Resets the environment by randomly giving each agent new positions and setting their 
        package status to False.
        """
        for agent in self.agents:
            new_x, new_y = self.generate_start()
            agent.has_package = False
            agent.rect.x = new_x
            agent.rect.y = new_y

    def render(self) -> None:
        """
        Renders the environment.
        """
        self.WINDOW.blit(self.FLOOR_TEXTURE, (0,0))
        for wall in self.level:
            self.WINDOW.blit(wall.texture, (wall.wall_element.x, wall.wall_element.y))
        
        for agent in self.agents:
            self.WINDOW.blit(agent.texture, (agent.rect.x, agent.rect.y))
        pygame.display.update()

    def generate_start(self) -> None:
        """
        Randomly generates an (x,y) pair of starting coordinates. Does not generate (x,y) that
        coincides with walls, pickup and delivery points.
        """
        x_size = len(self.LAYOUT[0])
        y_size = len(self.LAYOUT)
        collide = True

        while collide:
            x_pos = self.BLOCK_SIZE*random.randint(1, x_size - 1)
            y_pos = self.BLOCK_SIZE*random.randint(1, y_size - 1)

            for wall in self.level:
                if x_pos == wall.wall_element.x and y_pos == wall.wall_element.y:
                    collide = True
                    break

                collide = False

        return x_pos, y_pos

    def __iter__(self) -> None:
        return iter(self.agents)