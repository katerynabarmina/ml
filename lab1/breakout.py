import gymnasium as gym

# Create an environment for the Breakout game
env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")

# Display information about the action and observation spaces
print("Action Space:", env.action_space)  # Describes the range of possible actions
print("Observation Space:", env.observation_space)  # Describes the range and shape of observations

# Description of the Action Space
print("\nAvailable Actions:")
print(env.action_space.n)  # Total number of possible actions
print("Note: Each action corresponds to a specific move in the game (e.g., move left, move right, or hold the paddle).")

# Description of the Observation Space
print("\nObservation Space:")
print("Shape:", env.observation_space.shape)  # The dimensions of the observation space (e.g., image size)
print("Value Range:", env.observation_space.low.min(), "-", env.observation_space.high.max())  # Minimum and maximum pixel values

# Start the game
obs, info = env.reset()  # Reset the game to the initial state
done = False

while not done:
    # Take random actions in the game
    action = env.action_space.sample()  # Randomly sample an action from the action space
    obs, reward, done, truncated, info = env.step(action)  # Perform the action and update the environment
    print(f"Action: {action}, Reward: {reward}, Done: {done}")  # Log the action, reward, and game status

# Close the environment
env.close()
