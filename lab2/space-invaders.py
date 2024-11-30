import gymnasium as gym
import numpy as np
import time


def get_lowest_alien_position(aliens_positions):
    # Find the lowest alien (i.e., the alien closest to the spaceship vertically)
    closest_alien = None
    min_y = float('inf')

    for alien in aliens_positions:
        if alien[1] < min_y:  # Check if this alien is lower (closer to the player)
            min_y = alien[1]
            closest_alien = alien

    return closest_alien


def main():
    # Create the Space Invaders environment with the render mode specified
    env = gym.make("SpaceInvaders-v0", render_mode="human")  # Specify 'human' mode for visualization
    env.reset()

    done = False
    while not done:
        env.render()

        # Get the current observation (state)
        obs, reward, done, truncated, info = env.step(0)  # Placeholder action, will replace with strategy

        # Extract alien positions from the observation (example using RGB values, could vary)
        aliens_positions = []

        # For simplicity, we assume `obs` provides an array of positions of aliens (this will require image processing)
        # Here is a placeholder to simulate alien detection:
        for i in range(5):  # Assuming 5 aliens for simplicity
            aliens_positions.append((np.random.randint(0, 5), np.random.randint(0, 5)))  # (x, y) positions

        # Find the lowest alien (closest to spaceship)
        closest_alien = get_lowest_alien_position(aliens_positions)

        # Implementing the strategy:
        if closest_alien:
            alien_x, alien_y = closest_alien
            spaceship_x = 2  # Placeholder value (should be retrieved from the environment)

            # Move left if the alien is to the left
            if alien_x < spaceship_x:
                action = 4  # Action for moving left (in Space Invaders)
            # Move right if the alien is to the right
            elif alien_x > spaceship_x:
                action = 5  # Action for moving right
            # Fire if the alien is in line with the spaceship
            elif alien_x == spaceship_x:
                action = 1  # Action for shooting

            # Perform the action in the environment
            obs, reward, done, truncated, info = env.step(action)
            print(f"Action taken: {action}, Alien position: {alien_x, alien_y}, Reward: {reward}")

        time.sleep(0.05)  # Slow down for better visualization

    env.close()


if __name__ == "__main__":
    main()

