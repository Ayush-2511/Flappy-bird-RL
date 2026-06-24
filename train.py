import gymnasium as gym
from agent import Agent
import torch
import flappy_bird_gymnasium

env = gym.make("FlappyBird-v0", use_lidar=False)



agent = Agent()

episodes = 1000
batch_size = 256
episode = 0
while(agent.epsilon >= agent.epsilon_min):
    
    state,info = env.reset()
    done = False
    total_reward = 0
    while not done:

        action = agent.choose_action(state)
        next_state,reward,terminated,truncated,info = env.step(action)

        done = terminated or truncated

        agent.memory.push(
            state,
            action,
            reward,
            next_state,
            done
        )

        agent.learn(batch_size)
        state = next_state

        total_reward += reward
    episode+=1
    agent.epsilon = max(
        agent.epsilon_min, agent.epsilon*agent.epsilon_decay
    )

    if episode % 10 == 0:
        agent.update_target_network()

    print(
        f"Episode {episode}, "
        f"Reward: {total_reward}, "
        f"Epsilon: {agent.epsilon:.3f}"
    )

    if episode % 100 == 0:
        torch.save(
            agent.online_network.state_dict(),
            "flappy_dqn.pth"
        )

env.close()