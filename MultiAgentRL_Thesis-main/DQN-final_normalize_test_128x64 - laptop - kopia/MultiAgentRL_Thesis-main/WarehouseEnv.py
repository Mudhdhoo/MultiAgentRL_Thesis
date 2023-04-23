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
        self.CRASH_REWARD = -50
        self.PICKUP_REWARD = 1
        self.DELIVERY_REWARD = 2
        self.TERMINAL_REWARD = 0
        self.level = []
        self.make_level()
        self.starting_points = [120,40,120,80,120,120]
        #self.agents = [Agent(self.generate_starting_points(), block_size, self.AGENT_TEXTURE, self.NUM_AGENTS, gamma, batch_size, buffer_size,
        #          min_replay_size, eps_max, eps_min, eps_dec, update_freq, i, level_name) for i in range(1, num_agents + 1)]
        self.agents = [Agent((self.starting_points[2*(i-1)],self.starting_points[2*(i-1)+1]), block_size, self.AGENT_TEXTURE, self.NUM_AGENTS, gamma, batch_size, buffer_size,
                  min_replay_size, eps_max, eps_min, eps_dec, update_freq, i, level_name) for i in range(1, num_agents + 1)]
        for agent in self.agents:
            agent.communicate(self.agents)      # Initialize communication between agents

    def make_level(self) -> None:
        """
        Creates the level by building walls according to the given level layout.
        """
        p = 1
        d = 1
        x = 0
        y = 0
        for row in self.LAYOUT:
            for col in row:
                if col == "W":
                    self.level.append(WallElement((x,y), self.BLOCK_SIZE, self.WALL_TEXTURE, False))
                if col == "P":
                    self.level.append(WallElement((x,y), self.BLOCK_SIZE, self.PICKUP_TEXTURE, p))
                    p += 1
                if col == 'D':
                    self.level.append(WallElement((x,y), self.BLOCK_SIZE, self.DELIVERY_TEXTURE, d))
                    d += 1
                x += self.BLOCK_SIZE
            y += self.BLOCK_SIZE
            x = 0        

    def step(self) -> tuple:
        """
        Steps the environment forward one time-step according to the action taken by a specified
        agent. Returns the reward received by taking that action and the new state.
        """
        for agent in self.agents:
            if agent.done == False:
                agent.reward = self.STEP_REWARD
                #agent.done = False
                # Move agent
                agent.move(agent.action)
        for agent in self.agents:
            # Check for collisions with walls, pickup points and delivery points
            if agent.done == False:
                for wall in self.level:
                    if pygame.Rect.colliderect(agent.rect, wall.wall_element) and wall.texture == self.WALL_TEXTURE:
                        agent.reward = self.CRASH_REWARD
                        agent.crash = True
                        agent.crashes += 1
                        if agent.action == 0:
                            agent.move_right()
                        elif agent.action == 2: 
                            agent.move_left()
                        elif agent.action == 3:
                            agent.move_down()
                        elif agent.action == 1:
                            agent.move_up()

                    if pygame.Rect.colliderect(agent.rect, wall.wall_element) and wall.texture == self.PICKUP_TEXTURE and not agent.has_package and wall.index == agent.index: #((agent.index+1)%4+1):Replace 4 with self.NUM_AGENTS
                        agent.reward = self.PICKUP_REWARD
                        agent.has_package = True

                    if pygame.Rect.colliderect(agent.rect, wall.wall_element) and wall.texture == self.DELIVERY_TEXTURE and agent.has_package and wall.index == agent.index:
                        agent.reward = self.DELIVERY_REWARD
                        agent.done = True
                        agent.has_delivered = True
                        agent.delivered_packages += 1

                for other_agent in self.agents:
                    if pygame.Rect.colliderect(agent.rect, other_agent.rect) and agent != other_agent:
                        agent.reward = self.CRASH_REWARD
                        #other_agent.reward = self.CRASH_REWARD
                        agent.crash = True
                        #other_agent.done = True
                        agent.crashes += 1
                        #other_agent.crashes += 1
                        #return reward, agent.state, done
        for agent in self.agents:
            agent.communicate(self.agents)      # Get state information from other agents
        return
            #return reward, agent.state, done

    def reset(self) -> None:
        """
        Resets the environment by randomly giving each agent new positions and setting their 
        package status to False.
        """
        for agent in self.agents:
            #new_x, new_y = self.generate_starting_points()
            new_x, new_y = agent.starting_point[0], agent.starting_point[1]
            agent.has_package = False
            agent.rect.x = new_x
            agent.rect.y = new_y
            agent.done = False
            agent.crash = False
            agent.done_saved = False
            agent.position[1] = agent.rect.y
            agent.position[0] = agent.rect.x
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
    
    def init_buffer(self, min_replay_size): #Constructs and fills the buffer for the agent
        #state = self.state  #Saves initial state
        for i in range(min_replay_size): #Iterate as many times as the requiered memory amount
            done_check = []
            crash_check = []
            for agent in self.agents:
                agent.action = random.sample(agent.actions,1)[0] #Random action
                agent.old_state = agent.state
            self.step() #Step and save ifor
            for agent in self.agents:
                if agent.done_saved == False:
                    transition = (agent.old_state, agent.action, agent.reward, agent.done, agent.state) #To tuple
                    agent.replay_buffer.append(transition) #Save to memory
                    if agent.done == True:
                            agent.done_saved = True
                done_check.append(agent.done)
            #state = new_state #Update current state

            if False not in done_check or True in crash_check:
                self.reset() #If done reset