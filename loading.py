import numpy as np
import gym
from gym import wrappers
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from itertools import count
import torch.tensor as tensor
import argparse


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--phi',type=float,default=0,
                    help='Probability phi to perturb states')
parser.add_argument('--epsilon',type=float,default=1e-2,
                    help='Epsilon: the scaling factor to change the perturbed states')
args = parser.parse_args()

phi=args.phi
epsilon=args.epsilon


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
HIDDEN_LAYER = 64  # NN hidden layer size


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.saved_g = []
        self.saved_states = []
        self.saved_actions = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
if use_cuda:
    policy.cuda()
policy.load_state_dict(torch.load('./weights/Weights_'+str(phi)+'_'+str(epsilon)+'.pt'))
env = gym.make('CartPole-v0')

def select_action_eval(state):
    return policy(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)

def run_episode_eval( environment):
    state = environment.reset()
    steps = 0
    while True:
        # environment.render()
        state_tensor = torch.tensor([state],requires_grad=True)
        action = select_action_eval(state_tensor)
        next_state, reward, done, _ = environment.step(action.item())
        # Add perturbations here
        # negative reward when attempt ends
        
        if done:
            reward = -1

        state = next_state
        steps += 1
        if done:
            break
    return steps
iterate = 1000
a = 0
print(f'Evaluating for {iterate} number of episodes \n')
for i in range(iterate):
    tot_steps=run_episode_eval(env)
    a += tot_steps
print('Total steps taken (average): '+str(a//iterate))