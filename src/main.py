import pygame
import sys
import yaml
from pathlib import Path

class DisplayManager:
    def __init__(self):
        pygame.init()
        self.load_config()
        self.screen = pygame.display.set_mode(
            (self.config['display']['width'], 
             self.config['display']['height']),
            pygame.FULLSCREEN if self.config['display']['fullscreen'] else 0
        )
        pygame.display.set_caption("Digital Life")

    def load_config(self):
        config_path = Path(__file__).parent.parent / 'config' / 'display.yml'
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    pygame.quit()
                    sys.exit()

            # Clear screen
            self.screen.fill((0, 0, 0))  # Black background
            
            # Update display
            pygame.display.flip()

def main():
    display = DisplayManager()
    display.run()

if __name__ == "__main__":
    main()