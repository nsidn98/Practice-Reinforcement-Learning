import gym
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Discount factor. Model is not very sensitive to this value.
GAMMA = .95

# LR of 3e-2 explodes the gradients, LR of 3e-4 trains slower
LR = 3e-3
N_GAMES = 2000

N_STEPS = 200

env = gym.make('CartPole-v0')
N_ACTIONS = 2 # get from env
N_INPUTS = 4 # get from env


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.linear1 = nn.Linear(N_INPUTS, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)
        
        self.actor = nn.Linear(64, N_ACTIONS)
        self.critic = nn.Linear(64, 1)
    
    # In a PyTorch model, you only have to define the forward pass. PyTorch computes the backwards pass for you!
    def forward(self, x):
        x = self.linear1(x)
        x = F.tanh(x)
        x = self.linear2(x)
        x = F.tanh(x)
        x = self.linear3(x)
        x = F.tanh(x)
        return x
    
    # Only the Actor head
    def get_action_mean(self, x):
        x = self(x)
        action_mean = (self.actor(x))
        action_logstd = nn.parameters(torch.zeros_like(action_mean))
        action_std = torch.exp(action_logstd)
        return action_mean,action_std
    
    # Only the Critic head
    def get_state_value(self, x):
        x = self(x)
        state_value = self.critic(x)
        return state_value
    
    # Both heads
    def evaluate_actions(self, x):
        x = self(x)
        action_probs = F.softmax(self.actor(x))
        state_values = self.critic(x)
        return action_probs, state_values
        


protagonist = ActorCritic()
adversarial = ActorCritic()
optimizer_protagonist = optim.Adam(protagonist.parameters(), lr=LR)
optimizer_adversarial = optim.Adam(adversarial.parameters(), lr=LR)

state = env.reset()
finished_games = 0




def calc_actual_state_values(model,rewards, dones):
    R = []
    rewards.reverse()

    # If we happen to end the set on a terminal state, set next return to zero
    if dones[-1] == True: next_return = 0
        
    # If not terminal state, bootstrap v(s) using our critic
    # TODO: don't need to estimate again, just take from last value of v(s) estimates
    else:
        s = torch.from_numpy(states[-1]).float().unsqueeze(0)
        next_return = model.get_state_value(Variable(s)).data[0][0]
    
    # Backup from last state to calculate "true" returns for each state in the set
    R.append(next_return)
    dones.reverse()
    for r in range(1, len(rewards)):
        if not dones[r]: this_return = rewards[r] + next_return * GAMMA
        else: this_return = 0
        R.append(this_return)
        next_return = this_return

    R.reverse()
    state_values_true = Variable(torch.FloatTensor(R)).unsqueeze(1)
    
    return state_values_true


def reflect(model,optimizer,states, actions, rewards, dones):
    
    # Calculating the ground truth "labels" as described above
    state_values_true = calc_actual_state_values(model,rewards, dones)

    s = Variable(torch.FloatTensor(states))
    action_probs, state_values_est = model.evaluate_actions(s)
    action_log_probs = action_probs.log()
    
    a = Variable(torch.LongTensor(actions).view(-1,1))
    chosen_action_log_probs = action_log_probs.gather(1, a)

    # This is also the TD error
    advantages = state_values_true - state_values_est

    entropy = (action_probs * action_log_probs).sum(1).mean()
    action_gain = (chosen_action_log_probs * advantages).mean()
    value_loss = advantages.pow(2).mean()
    total_loss = value_loss - action_gain - 0.0001*entropy

    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), 0.5)
    optimizer.step()
    
    
def test_model(model):
    score = 0
    done = False
    env = gym.make('CartPole-v0')
    state = env.reset()
    global action_probs
    while not done:
        score += 1
        s = torch.from_numpy(state).float().unsqueeze(0)
        
        action_probs = model.get_action_probs(Variable(s))
        
        _, action_index = action_probs.max(1)
        action = action_index.data[0]
        next_state, reward, done, thing = env.step(action.item())
        state = next_state
    return score
    
    
while finished_games < N_GAMES:
    states, actions_protagonist,actions_adversarial, rewards, dones = [], [], [], [], []

    # Gather training data
    for i in range(N_STEPS):
        s = Variable(torch.from_numpy(state).float().unsqueeze(0))

        #protagonist
        action_mean_protagonist,action_std_protagonist = protagonist.get_action_mean(s)
        action_protagonist = torch.normal(action_mean_protagonist[0],action_std_protagonist[0]).data
        
        #adversaial
        action_mean_adversarial,action_std_adversarial = adversarial.get_action_probs(s)
        action_adversarial = torch.normal(action_mean_adversarial[0],action_std_adversarial[0]).data
        
        #take mean of adversarial action
        action = (action_adversarial+action_protagonist).mean()
        next_state, reward, done, _ = env.step(action.item())

        states.append(state); actions.append(action); rewards.append(reward); dones.append(done)

        if done:
            state = env.reset()
            finished_games += 1
        else:
            state = next_state

    # Reflect on training data
    reflect(protagonist,optimizer_protagonist,states, actions, rewards, dones)
    reflect(adversaial,optimizer_adversarial,states, actions, -rewards, dones)
    if finished_games%100==0:
        print(f'Finished {finished_games} games')
        
    
print('Testing model')
tot=0
iterate=1000
for i in tqdm(range(iterate)):
    a=test_model(model)
    tot+=a
print(f'Final score:{tot/iterate}')