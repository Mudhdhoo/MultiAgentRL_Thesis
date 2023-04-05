import pygame

class WallElement:
    def __init__(self, wall_pos:tuple, wall_size:tuple, texture) -> None:
        self.wall_element = pygame.Rect(wall_pos[0], wall_pos[1], wall_size, wall_size)
        self.texture = texture
