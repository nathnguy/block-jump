# Block Jump - game ends when the player block hits an obstacle block
# Press space to jump!

import pygame
from pygame import Rect
import random

pygame.init()

font = pygame.font.SysFont("arial", 20)
WINDOW_SIZE = (256, 256)
BACKGROUND_COLOR = (255, 255, 255)
PLAYER_COLOR = (75, 75, 75)
OBS_COLOR = (150, 150, 150)

FPS = 30

PLAYER_SIZE = 30

# world positions
GROUND_OFFSET = 180
PLAYER_LEFT_PADDING = 20

JUMP_VELOCITY = 650
OBS_VELOCITY = 250
JUMP_HEIGHT = 90
GRAVITY = 2050

# obstacle dimension bounds
OBS_OFFSET_MIN = 150
OBS_OFFSET_MAX = 250
OBS_WIDTH_MIN = 10
OBS_WIDTH_MAX = 50
OBS_HEIGHT_MIN = 40
OBS_HEIGHT_MAX = 75

# only generate the largest obstacle
# OBS_WIDTH_MIN = 50
# OBS_WIDTH_MAX = 50
# OBS_HEIGHT_MIN = 75
# OBS_HEIGHT_MAX = 75

NUM_INITIAL_OBS = 5
OBS_INITIAL_X = 100

# actions chosen by training agent
ACTION_JUMP = 0

class Game():

    def __init__(self, train=False):
        # set window preferences
        self.window = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Block Jump")
        self.train = train
        self.reset()

    
    def reset(self):
        self.clock = pygame.time.Clock()
        
        # player position
        self.player_y = 0
        self.player_velocity_y = 0

        # obstacles
        self.obstacles = []
        x = OBS_INITIAL_X
        width = 0
        for _ in range(NUM_INITIAL_OBS):
            x, y, width, height = self._generate_obstacle(x + width)
            self.obstacles.append(Rect(x, y, width, height))

        self.score = 0


    # prev_x - position to offset from 
    # returns: x, y, width, and height for pygame.Rect
    def _generate_obstacle(self, prev_x):
        x = prev_x + random.randint(OBS_OFFSET_MIN, OBS_OFFSET_MAX)
        width = random.randint(OBS_WIDTH_MIN, OBS_WIDTH_MAX)
        height = random.randint(OBS_HEIGHT_MIN, OBS_HEIGHT_MAX)
        y = GROUND_OFFSET - height

        return x, y, width, height

    
    def draw(self):
        self.window.fill(BACKGROUND_COLOR)

        # player
        pygame.draw.rect(self.window, PLAYER_COLOR, 
                         Rect(PLAYER_LEFT_PADDING, # x pos
                              GROUND_OFFSET - self.player_y - PLAYER_SIZE, # y pos
                              PLAYER_SIZE, PLAYER_SIZE)) # rect size

        # obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.window, OBS_COLOR, obs)

        pygame.display.flip()


    # returns:
    #   - game_over: true iff game is over, false otherwise
    #   - reward: +1 living reward
    def update(self):
        dt = self.clock.tick(FPS) / 1000

        # player input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # See if the user clicks the red x 
                pygame.quit()  # Quit the game
                quit()

            if event.type == pygame.KEYDOWN and not self.train:
                if event.key == pygame.K_SPACE and self.player_velocity_y == 0:
                    # jump
                    self.player_velocity_y = JUMP_VELOCITY

        # player update

        self.player_y += self.player_velocity_y * dt

        if self.player_y <= 0:
            self.player_velocity_y = 0
            self.player_y = 0
        else:
            self.player_velocity_y -= GRAVITY * dt

        # obstacle update
        player_rect = Rect(PLAYER_LEFT_PADDING, GROUND_OFFSET - self.player_y - PLAYER_SIZE,
                           PLAYER_SIZE, PLAYER_SIZE)

        num_passed = 0
        for i, obs in enumerate(self.obstacles):
            x, _, width, _ = obs
            self.obstacles[i] = obs.move(-OBS_VELOCITY * dt, 0)
            if self.obstacles[i].colliderect(player_rect):
                return True, 0
            elif x < -width:
                num_passed += 1

        for _ in range(num_passed):
            x, _, width, _ = self.obstacles[-1]
            self.obstacles.pop(0)
            self.obstacles.append(Rect(*self._generate_obstacle(x + width)))

        self.score += 1 * dt

        return False, 1

    
    # state representation for the game is the entire window
    def grayscale_image(self):
        width = self.window.get_width()
        height = self.window.get_height()

        image = []

        for row in range(height):
            image.append([])
            for col in range(width):
                pixel = self.window.get_at((col, row))
                image[row].append(pixel.r / 255)

        return [image]


    # for letting model select actions
    def move(self, action):
        if action == ACTION_JUMP and self.player_velocity_y == 0:
            # jump
            self.player_velocity_y = JUMP_VELOCITY



if __name__ == "__main__":
    game = Game()

    while True:
        game.draw()
        game_over, _ = game.update()

        if game_over:
            print(f"Final Score: {game.score: .3f}")
            game.reset()
