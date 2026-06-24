import random
import torch
import torch.optim as optim

from model import DQN
from replay_buffer import ReplayBuffer

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)

class Agent:

    def __init__(self):
        self.online_network = DQN().to(device)
        self.target_network = DQN().to(device)
        self.memory = ReplayBuffer(500000)

        self.optimizer = optim.Adam(
            self.online_network.parameters(),
            lr=0.005
        )

        self.gamma = 0.995
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999

    def choose_action(self, state):

        if random.random() < self.epsilon:
            return random.randint(0,1)
        
        state = torch.tensor(
            state,
            dtype=torch.float32,
            device=device
        )

        with torch.no_grad():
            q_values = self.online_network(state)

        return torch.argmax(q_values).item()
    
    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        batch = self.memory.sample(batch_size)
        states,actions,rewards,next_states,dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        current_q = self.online_network(states)

        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_network(next_states)
            max_next_q = next_q.max(dim=1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = torch.nn.functional.mse_loss(
            current_q,
            target_q
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )