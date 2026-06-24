import gymnasium as gym
import torch
from agent import Agent
import agent as ag
import flappy_bird_gymnasium

env = gym.make("FlappyBird-v0",render_mode="human", use_lidar=False)

agent = Agent()

agent.online_network.load_state_dict(
    torch.load("flappy_dqn.pth", map_location=ag.device),
)

agent.epsilon = 0

episodes = 20

for episode in range(episodes):

    state,info = env.reset()

    done = False
    total_reward = 0

    while not done:
        action = agent.choose_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        state = next_state

        total_reward += reward

    print(
        f"Test Episode {episode}: Reward = {total_reward}"
    )

env.close()