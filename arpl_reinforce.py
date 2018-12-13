import argparse
import gym
import numpy as np
from itertools import count


# use hooks
'''
grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

x = Variable(torch.randn(1,1), requires_grad=True)
y = 3*x
z = y**2

# In here, save_grad('y') returns a hook (a function) that keeps 'y' as name
y.register_hook(save_grad('y'))
z.register_hook(save_grad('z'))
z.backward()

print(grads['y'])
print(grads['z'])
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--phi',type=float,default=0,
                    help='Probability phi to perturb states')
parser.add_argument('--epsilon',type=float,default=1e-2,
                    help='Epsilon: the scaling factor to change the perturbed states')
parser.add_argument('--threshold',type=int,default=185,
                    help='Threshold for rewards to stop the iterations for cartpole')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


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
# if use_cuda:
#     policy.cuda()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()

grad_state = {}
def save_grad(name):
    def hook(grad):
        grad_state[name] = grad
    return hook

def select_action(i,state):
    state = torch.tensor(state,requires_grad=True).float().unsqueeze(0)
    # print(state)
    state.register_hook(save_grad(str(i)))
    probs = policy(state)
    # print(probs[0,1])
    # print(probs)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    policy.saved_states.append(state)
    policy.saved_actions.append(action)
    return action.item()

def finish_episode():
    R = 0
    policy_loss = []
    rewards = torch.tensor(policy.saved_g)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward(retain_graph=True)
    # print(grad_state['1'])
    
    arpl_perturb() ###Adding pertubations here
    del policy_loss

    policy_loss_per = []
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss_per.append(-log_prob * reward)
    policy_loss_per = torch.cat(policy_loss_per).sum()
    optimizer.zero_grad()
    policy_loss_per.backward()
    optimizer.step()

    del policy_loss_per
    
    del policy.saved_g[:]
    del policy.saved_actions[:]
    del policy.saved_states[:]
    del policy.saved_log_probs[:]
    grad_state.clear()


def arpl_perturb(phi=args.phi,epsilon=args.epsilon):
    for t,s in enumerate(policy.saved_states):
        flag = np.float(np.random.choice([0,1],p=[1-phi,phi])) # to perturb with probability phi
        if flag:
            perturbed_state = s + epsilon*grad_state[str(t)]
            policy.saved_states[t] = perturbed_state
            probs = policy(perturbed_state)
            log_prob = torch.log(probs[0,policy.saved_actions[t]])
            policy.saved_log_probs[t] = log_prob
        
def main():
    running_reward = 10
    for i_episode in count(1):
        rewards = []
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(t,state)
            state, reward, done, _ = env.step(action)
            if done:
                reward = -1
            if args.render:
                env.render()
            rewards.append(reward)
            if done:
                break
        
        # compute g
        R = 0
        for r in rewards[::-1]:
            R = r + args.gamma * R
            policy.saved_g.insert(0, R)


        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > args.threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break
    torch.save(policy.state_dict(),'Weights_'+str(args.phi)+'_'+str(args.epsilon)+'.pt')
    torch.save(optimizer.state_dict(),'Optimizer_'+str(args.phi)+'_'+str(args.epsilon)+'.pt')


if __name__ == '__main__':
    main()