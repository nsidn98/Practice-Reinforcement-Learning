# Solution of Open AI gym environment "Cartpole-v0" (https://gym.openai.com/envs/CartPole-v0) using DQN and Pytorch.
# It is is slightly modified version of Pytorch DQN tutorial from
# http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html.
# The main difference is that it does not take rendered screen as input but it simply uses observation values from the \
# environment.

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

# hyper parameters
EPISODES = 200  # number of episodes
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.05  # e-greedy threshold end value
EPS_DECAY = 200  # e-greedy threshold decay
GAMMA = 0.8  # Q-learning discount factor
LR = 0.001  # NN optimizer learning rate
HIDDEN_LAYER = 64  # NN hidden layer size
BATCH_SIZE = 64  # Q-learning batch size
iterate=50
# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def latest_state(self):
        return self.memory[-1]
        
    def overwrite(self,arr):
        self.memory[-1]=arr


class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


env = gym.make('CartPole-v0')

model = Network()
if use_cuda:
    model.cuda()
memory = ReplayMemory(10000)
optimizer = optim.Adam(model.parameters(), LR)
steps_done = 0
episode_durations = []


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        return model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(2)]])

phi=0.4
epsilon=1

def select_action_eval(state):
    return model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)

def run_episode(e, environment,optimizer):
    state = environment.reset()
    steps = 0
    while True:
        # environment.render()
        state_tensor = torch.tensor([state],requires_grad=True)
        action = select_action(state_tensor)
        print(state_tensor)
        next_state, reward, done, _ = environment.step(action.item())
        # Add perturbations here
        # negative reward when attempt ends
        
        if done:
            reward = -1

        memory.push((FloatTensor([state]),
                     action,  # action is already a tensor
                     FloatTensor([next_state]),
                     FloatTensor([reward])))

        flag = np.float(np.random.choice([0,1],p=[1-phi,phi])) # to perturb with probability phi
        state_grad = learn(flag,epsilon,optimizer)
        if state_grad is not None:
        #### have to add perturbation here ####
        ## if flag == 0 then don't perturb, else perturb
        # state_grad = state_tensor.grad
            # print(f'StateGrad:{state_grad}')
            delta = flag*epsilon*state_grad
            d = next_state
            next_state = next_state + delta
            memory.overwrite((FloatTensor([state]),
                        action,  # action is already a tensor
                        FloatTensor([next_state]),
                        FloatTensor([reward])))
            # print(flag)
            # print(f'Original State:{d} \n Changed state:{next_state}')
        state = next_state
        steps += 1
        if done:
            print("{2} Episode {0} finished after {1} steps"
                  .format(e, steps, '\033[92m' if steps >= 195 else '\033[99m'))
            episode_durations.append(steps)
            # avg_rew = avg_rew * 0.99 + steps * 0.01
            break
    return

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

def exp_lr_scheduler(optimizer,init_lr,epoch,lr_decay=2):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def learn(flag,epsilon,optimizer):
    if len(memory) < BATCH_SIZE:
        return

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(BATCH_SIZE)
    transitions.append(memory.latest_state())   # to make sure the latest transition is there in the array
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)
    
    batch_state = tensor(torch.cat(batch_state),requires_grad=True)
    batch_action = tensor(torch.cat(batch_action))
    batch_reward = tensor(torch.cat(batch_reward))
    batch_next_state = tensor(torch.cat(batch_next_state))

    # current Q values are estimated by NN for all actions
    current_q_values = model(batch_state).gather(1, batch_action)
    # expected Q values are estimated from actions which gives maximum Q value
    max_next_q_values = model(batch_next_state).detach().max(1)[0]
    expected_q_values = batch_reward + (GAMMA * max_next_q_values)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(current_q_values[:,0], expected_q_values)

   # backpropagation of loss to NN
    loss.backward()
    state_grad=batch_state.grad[-1]

    batch_next_state[-1] = batch_next_state[-1] + epsilon*flag*state_grad

    current_q_values = model(batch_state).gather(1, batch_action)
    # expected Q values are estimated from actions which gives maximum Q value
    max_next_q_values = model(batch_next_state).detach().max(1)[0]
    expected_q_values = batch_reward + (GAMMA * max_next_q_values)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(current_q_values[:,0], expected_q_values)
    # batch_state.grad[-1]
    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    # print(f'grad:{state_grad}')
    optimizer.step()

    state_grad = [t.item() for t in state_grad]
    return np.array(state_grad)

running_reward = 10
flag_thresh = 0
for e in count(1):
    run_episode(e,env,optimizer)
    if e%10==0:
        a = 0
        for i in range(iterate):
            tot_steps=run_episode_eval(env)
            a += tot_steps
        print('Tot_steps: '+str(a//iterate))
        if a//iterate>195:
            flag_thresh +=1
            optimizer = exp_lr_scheduler(optimizer,LR,flag_thresh)
    if flag_thresh>=10:
        break
torch.save(model.state_dict(),"weights0_4.pt")
torch.save(optimizer.state_dict(),"optim0_4.pt")
iterate = 1000
a = 0
for i in range(iterate):
    tot_steps=run_episode_eval(env)
    a += tot_steps
print('Tot_steps: '+str(a//iterate))


from graphviz import Digraph
import re
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Variable
import torchvision.models as models


def make_dot(var):
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def add_nodes(var):
        if var not in seen:
            if isinstance(var, Variable):
                value = '('+(', ').join(['%d'% v for v in var.size()])+')'
                dot.node(str(id(var)), str(value), fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'previous_functions'):
                for u in var.previous_functions:
                    dot.edge(str(id(u[0])), str(id(var)))
                    add_nodes(u[0])
    add_nodes(var.creator)
    return dot


inputs = torch.randn(4,requires_grad=True)
model = policy(inputs)
m = Categorical(probs)
action = m.sample()

g = make_dot(action)
print(g)