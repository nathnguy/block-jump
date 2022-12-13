# Reinforcement learning agent that chooses actions
# and remembers state, action, and reward info
# source: https://www.youtube.com/watch?v=wc-FxNENg9U

from model import BlockNet
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_TARGET_NET = False

# Hyper Parameters:
GAMMA = 0.9 # discount factor
EPSILON = 1 # take random vs predicted action
EPSILON_MIN = 0.001
BATCH_SIZE = 32
TAU = 0.0005 # how much to update target net

# 0.001 - no target net
LR = 0.001

MAX_MEM = 100_000
NUM_ACTIONS = 2
IMAGE_SIZE = (1, 256, 256) # CHW

# we decrease epsilon evenly for NUM_GAMES
# to go from EPSILON to EPSILON_MIN
NUM_GAMES = 500

class Agent():
    
    def __init__(self):
        self.epsilon = EPSILON
        self.action_space = [i for i in range(NUM_ACTIONS)]
        self.mem_counter = 0

        self.model = BlockNet() # online net
        self.model.train()

        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.loss = nn.MSELoss()

        if USE_TARGET_NET:
            self.target_net = BlockNet()
            self.target_net.load_state_dict(self.model.state_dict())
            self.target_net.train()

        # replay memory
        self.state_memory = np.zeros((MAX_MEM, *IMAGE_SIZE), dtype=np.float32)
        self.new_state_memory = np.zeros((MAX_MEM, *IMAGE_SIZE), dtype=np.float32)
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


    # update Q-values, replay memory
    def learn(self):
        if self.mem_counter < BATCH_SIZE:
            return

        self.optimizer.zero_grad()

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

        if not USE_TARGET_NET:
            q_next = self.model.forward(new_state_batch)
        else:
            with torch.no_grad():
                q_next = self.target_net.forward(new_state_batch)

        q_next[terminal_batch] = 0.0

        q_target = reward_batch + GAMMA * torch.max(q_next, dim=1)[0]

        loss = self.loss(q_target, q_eval).to(self.model.device)
        loss.backward()
        self.optimizer.step()

        # soft update target net
        if USE_TARGET_NET:
            target_state_dict = self.target_net.state_dict()
            model_state_dict = self.model.state_dict()
            for key in model_state_dict:
                target_state_dict[key] = model_state_dict[key] * TAU + target_state_dict[key] * (1 - TAU)
            self.target_net.load_state_dict(target_state_dict)

        return loss.item()


    # go from 1 to EPSILON_MIN in NUM_GAMES
    def decay_epsilon(self):
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= (EPSILON - EPSILON_MIN) / NUM_GAMES
        else:
            self.epsilon = EPSILON_MIN