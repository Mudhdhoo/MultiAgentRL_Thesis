import pygame

class Agent:
    def __init__(self, init_pos:tuple, size:NotImplementedError, texture) -> None:
        self.rect = pygame.Rect(init_pos[0], init_pos[1], size, size)
        self.size = size
        self.texture = texture

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

    def move_down(self):
        self.rect.y += self.size

    def move_left(self):
        self.rect.x -= self.size

    def move_right(self):
        self.rect.x += self.size
