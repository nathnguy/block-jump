# Reinforcement learning agent that chooses actions
# and remembers state, action, and reward info
# source: https://www.youtube.com/watch?v=wc-FxNENg9U

from model import LR, BlockNet
import numpy as np
import torch

# Hyper Parameters:
GAMMA = 0.9 # discount factor
EPSILON = 1 # take random vs predicted action
EPSILON_MIN = 0.001
BATCH_SIZE = 32

MAX_MEM = 100_000
NUM_ACTIONS = 2
IMAGE_SIZE = (256, 256)

# we decrease epsilon evenly for NUM_GAMES
# to go from EPSILON to EPSILON_MIN
NUM_GAMES = 200

TARGET_UPDATE = 100

class Agent():
    
    def __init__(self):
        self.epsilon = EPSILON
        self.action_space = [i for i in range(NUM_ACTIONS)]
        self.mem_counter = 0

        self.model = BlockNet() # online net
        self.model.train()

        # TODO
        # self.target_net = BlockNet()
        # self.target_net.load_state_dict(self.model.state_dict())
        # self.target_net.train()

        # replay memory
        self.state_memory = np.zeros((MAX_MEM, 1, *IMAGE_SIZE), dtype=np.float32)
        self.new_state_memory = np.zeros((MAX_MEM, 1, *IMAGE_SIZE), dtype=np.float32)
        self.action_memory = np.zeros(MAX_MEM, dtype=np.int32)
        self.reward_memory = np.zeros(MAX_MEM, dtype=np.float32)
        self.terminal_memory = np.zeros(MAX_MEM, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_next, game_over):
        index = self.mem_counter % MAX_MEM
        self.state_memory[index] = state
        self.new_state_memory[index] = state_next
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = game_over

        self.mem_counter += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # take predicted action
            state = torch.tensor(np.array([observation])).to(self.model.device)
            actions = self.model.forward(state)
            action = torch.argmax(actions).item()
        else:
            # take random action
            action = np.random.choice(self.action_space)

        return action

    # update Q-values
    def learn(self):
        if self.mem_counter < BATCH_SIZE:
            return

        self.model.optimizer.zero_grad()

        mem_size = min(self.mem_counter, MAX_MEM)
        batch = np.random.choice(mem_size, BATCH_SIZE, replace=False)
        batch_index = np.arange(BATCH_SIZE, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.model.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.model.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.model.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.model.device)

        action_batch = self.action_memory[batch]

        # q_eval: 
        # select all batch elements with batch_index
        # pick Q-value of action specified by action_batch
        q_eval = self.model.forward(state_batch)[batch_index, action_batch]
        q_next = self.model.forward(new_state_batch)
        # q_next = self.target_net.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + GAMMA * torch.max(q_next, dim=1)[0]

        loss = self.model.loss(q_target, q_eval).to(self.model.device)
        loss.backward()
        self.model.optimizer.step()

        # update target net
        # if self.mem_counter % TARGET_UPDATE == 0:
        #     self.target_net.load_state_dict(self.model.state_dict())

        return loss.item()

    # go from 1 to EPSILON_MIN in NUM_GAMES
    def decay_epsilon(self):
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= (EPSILON - EPSILON_MIN) / NUM_GAMES