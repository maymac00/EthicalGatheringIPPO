import pygame
import sys
import numpy as np
screen = None


class GatheringDisplay:
    PLAYER_1_SPRITE = pygame.image.load('utils/sprites/player1.png')
    PLAYER_2_SPRITE = pygame.image.load('utils/sprites/player2.png')
    APPLE_SPRITE = pygame.image.load('utils/sprites/apple.png')
    GROUND_SPRITE = pygame.image.load('utils/sprites/Grass16.png')
    text_color = (0, 0, 0)

    def __init__(self):
        self.screen = None
        self.clock = None
        self.font = None

    def draw_grid(self, matrix, text, tile_size=100):

        # Set up the display
        matrix_height = len(matrix)
        matrix_width = len(matrix[0])

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((matrix_width * tile_size, matrix_height * tile_size))
            pygame.display.set_caption("Gathering Game")
            self.font = pygame.font.Font(None, 24)  # You can use a custom font by replacing None with a path to a .ttf file
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Draw the grid world
        for y in range(matrix_height):
            for x in range(matrix_width):
                element = matrix[y][x]
                position = (x * tile_size, y * tile_size)

                # Draw the ground sprite first
                ground_sprite = pygame.transform.scale(GatheringDisplay.GROUND_SPRITE, (tile_size, tile_size))
                self.screen.blit(ground_sprite, position)

                # Draw other elements on top of the ground
                if element == 'A':  # Player 1
                    sprite = GatheringDisplay.PLAYER_1_SPRITE
                elif element == 'B':  # Player 2
                    sprite = GatheringDisplay.PLAYER_2_SPRITE
                elif element == '@':  # Apple
                    sprite = GatheringDisplay.APPLE_SPRITE
                else:  # No element, continue to the next cell
                    continue

                sprite = pygame.transform.scale(sprite, (tile_size, tile_size))
                self.screen.blit(sprite, position)

            # Display the text with a white background
            text_surface = self.font.render(text, True, GatheringDisplay.text_color, (255, 255, 255))
            text_width, text_height = text_surface.get_size()

            screen_width, screen_height = self.screen.get_size()

            # Calculate the bottom center position
            margin_bottom = 0  # You can adjust the margin as needed
            text_x = (screen_width - text_width) // 2
            text_y = screen_height - text_height - margin_bottom
            text_position = (text_x, text_y)
            self.screen.blit(text_surface, text_position)
            # Update the display
            pygame.display.flip()
