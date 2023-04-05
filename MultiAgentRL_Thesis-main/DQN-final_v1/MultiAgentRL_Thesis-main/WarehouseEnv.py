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

    def __init__(self, layout:list, block_size:int, num_agents:int, gamma, batch_size, buffer_size,
                  min_replay_size, eps_max, eps_min, eps_dec, update_freq, level_name) -> None:
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
        self.TERMINAL_REWARD = 0
        self.level = []
        self.make_level()
        self.agents = [Agent(self.generate_starting_points(), block_size, self.AGENT_TEXTURE, self.NUM_AGENTS, gamma, batch_size, buffer_size,
                  min_replay_size, eps_max, eps_min, eps_dec, update_freq, i, level_name) for i in range(1, num_agents + 1)]
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
        done = False
        # Move agent
        agent.move(button_pressed)

        # Check for collisions with walls, pickup points and delivery points
        for wall in self.level:
            if pygame.Rect.colliderect(agent.rect, wall.wall_element) and wall.texture == self.WALL_TEXTURE:
                reward = self.CRASH_REWARD
                done = True
                agent.crashes += 1
                if button_pressed == 0:
                    agent.move_right()
                elif button_pressed == 2: 
                    agent.move_left()
                elif button_pressed == 1:
                    agent.move_down()
                elif button_pressed == 3:
                    agent.move_up()
                return reward, agent.state, done

            if pygame.Rect.colliderect(agent.rect, wall.wall_element) and wall.texture == self.PICKUP_TEXTURE and not agent.has_package:
                reward = self.PICKUP_REWARD
                agent.has_package = True

            if pygame.Rect.colliderect(agent.rect, wall.wall_element) and wall.texture == self.DELIVERY_TEXTURE and agent.has_package:
                reward = self.DELIVERY_REWARD
                done = True
                agent.has_delivered = True
                agent.delivered_packages += 1
                #return reward, agent.state, done #La till detta

        for other_agent in self.agents:
            if pygame.Rect.colliderect(agent.rect, other_agent.rect) and agent != other_agent:
                reward = self.CRASH_REWARD
                done = True
                agent.crashes += 1
                other_agent.crashes += 1
                return reward, agent.state, done
        for agent in self.agents:
            agent.communicate(self.agents)      # Get state information from other agents

        return reward, agent.state, done

    def reset(self) -> None:
        """
        Resets the environment by randomly giving each agent new positions and setting their 
        package status to False.
        """
        for agent in self.agents:
            new_x, new_y = self.generate_starting_points()
            agent.has_package = False
            agent.rect.x = new_x
            agent.rect.y = new_y
            agent.done = False
        for agent in self.agents:
            agent.communicate(self.agents)

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

    def generate_starting_points(self) -> None:
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