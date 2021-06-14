import argparse
from itertools import count

import gym
import gym.spaces
import gym_panda
import scipy.optimize
import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
#from loss import *
import time
import pdb

import pickle

def train_inverse_dynamics(feasible_traj_path, inverse_model, state_dim, action_dim, num_epochs=10, bs=128):
    #pdb.set_trace()
    trajs = pickle.load(open(feasible_traj_path[0], 'rb'))  #trajs is list of [s,s',a]
    all_tuples = np.array(trajs)
    training = inverse_model.training
    inverse_model.train()
    for ii in range(num_epochs):
        order = np.random.permutation(len(all_tuples))
        loss_epoch = 0
        for jj in range((len(all_tuples)-1)//bs+1):
            idx = order[jj*bs:(jj+1)*bs]
            sampled_batch = all_tuples[idx]
            input_ = torch.Tensor(sampled_batch).float().to(device)
            inverse_optimizer.zero_grad()
            output_action = inverse_model(torch.cat([input_[:,0:state_dim], input_[:,state_dim:state_dim*2]], dim=1))
            loss = nn.SmoothL1Loss()(output_action, input_[:,-action_dim:].detach())
            loss.backward()
            inverse_optimizer.step()
            loss_epoch += loss.item()
        print('inverse loss', loss_epoch/((len(all_tuples)-1)//bs+1))
    inverse_model.train(training)


def train_inverse_forward_dynamic(num_epochs, feasible_traj, inverse_model, forward_model, action_dim, bs=128):
    inverse_training = inverse_model.training
    inverse_model.train()
    forward_training = forward_model.training
    forward_model.train()
    state_dim = (feasible_traj.shape[1]-action_dim)//2
    for ii in range(num_epochs):
        order = np.random.permutation(feasible_traj.shape[0])
        loss_epoch = 0
        for jj in range((len(feasible_traj)-1)//bs+1):
            idx = order[jj*bs:(jj+1)*bs]
            sampled_batch = feasible_traj[idx]
            if 'action1' in args.env:
                sampled_batch[:,-2] = np.clip(sampled_batch[:,-action_dim], -1, 0.)
                sampled_batch[:,-1] = 0.
            elif 'action2' in args.env:
                sampled_batch[:,-2] = np.clip(sampled_batch[:,-action_dim], 0, 1.)
                sampled_batch[:,-1] = 0.
            sampled_batch = torch.Tensor(sampled_batch).float().to(device)
            inverse_optimizer.zero_grad()
            forward_optimizer.zero_grad()
            output_action = inverse_model(sampled_batch[:,0:-action_dim])
            output_state = forward_model(torch.cat([sampled_batch[:,0:state_dim], sampled_batch[:,-action_dim:]], dim=1))
            back_state = forward_model(torch.cat([sampled_batch[:,0:state_dim], output_action], dim=1))
            back_action = inverse_model(torch.cat([sampled_batch[:,0:state_dim], output_state], dim=1))
            inverse_loss = nn.SmoothL1Loss()(output_action, sampled_batch[:,-action_dim:].detach())
            forward_loss = nn.SmoothL1Loss()(output_state+sampled_batch[:,0:state_dim], sampled_batch[:,state_dim:state_dim*2].detach())
            state_consistency_loss = nn.SmoothL1Loss()(back_state+sampled_batch[:,0:state_dim], sampled_batch[:,state_dim:state_dim*2].detach())
            action_consistency_loss = nn.SmoothL1Loss()(back_action, sampled_batch[:,-action_dim:].detach())
            loss = inverse_loss + forward_loss + state_consistency_loss + action_consistency_loss
            loss.backward()
            inverse_optimizer.step()
            forward_optimizer.step()
            loss_epoch += loss.item()
        print(output_action[-1], sampled_batch[:,-action_dim:].detach()[-1])
        print('inverse loss', loss_epoch/((len(feasible_traj)-1)//bs+1))
    inverse_model.train(inverse_training)
    forward_model.train(forward_training)


torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', type=str, default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=1111, metavar='N',
                    help='random seed (default: 1111')
parser.add_argument('--batch-size', type=int, default=5000, metavar='N',
                    help='size of a single batch')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--eval-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--num-epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train an expert')
parser.add_argument('--hidden-dim', type=int, default=64, metavar='H',
                    help='the size of hidden layers')
parser.add_argument('--lr', type=float, default=1e-3, metavar='L',
                    help='learning rate')
parser.add_argument('--vf-iters', type=int, default=30, metavar='V',
                    help='number of iterations of value function optimization iterations per each policy optimization step')
parser.add_argument('--vf-lr', type=float, default=3e-4, metavar='V',
                    help='learning rate of value network')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--xml', default=None, help='the xml configuration file')
parser.add_argument('--demo_files', nargs='+', help='the environment used for test')
parser.add_argument('--ratios', nargs='+', type=float, help='the ratio of demos to load')
parser.add_argument('--eval_epochs', type=int, default=10, help='the epochs for evaluation')
parser.add_argument('--save_path', help='the path to save model')
parser.add_argument('--feasible_traj_paths', nargs='+')
parser.add_argument('--mode')
args = parser.parse_args()

log_file = open(args.save_path.split('/')[-1].split('.pth')[0]+'_seed_{}.txt'.format(args.seed), 'w')
save_path = args.save_path.replace('.pth', '_seed_{}.pth'.format(args.seed))
#pdb.set_trace()
env = gym.make(args.env_name)
f_env = gym.make(args.env_name)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

def load_demos(demo_files):
    state_files = []
    trajs = []
    traj_id = 0
    pair_traj_id = []
    raw_demos = pickle.load(open(demo_files[0], 'rb'))  #demo 5+48
    #pdb.set_trace()
    trajs = raw_demos
    state_pairs = []
    for j in range(len(trajs)):
        #state_pairs = []
        for i in range(len(trajs[j])-1):
            state_pairs.append(np.concatenate([np.array(trajs[j][i]), np.array(trajs[j][i+1])], axis=0))
        pair_traj_id.append(np.array([traj_id]*np.array(trajs[j]).shape[0]))
        traj_id += 1
        #state_files.append(np.concatenate(state_pairs, axis=0))
        state_files.append(np.array(state_pairs))
    #pdb.set_trace()
    return state_pairs, trajs, np.concatenate(pair_traj_id, axis=0)

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def compute_feasibility_pair(expert_trajs, model, f_env):
    all_distance = []
    for index in range(len(expert_trajs)):
        expert_traj = expert_trajs[index]
        batch_size = 64
        batch_num = (expert_traj.shape[0]-1)//batch_size + 1
        with torch.no_grad():
            for i in range(batch_num):
                f_env.reset()
                action = model(torch.from_numpy(expert_traj[i*batch_size:(i+1)*batch_size]).float().to(device)).cpu().numpy()
                next_states = []
                for j in range(action.shape[0]):
                    f_env.set_observation(expert_traj[i*batch_size+j])
                    next_state, _, _, _ = f_env.step(action[j])
                    next_states.append(next_state)
                next_states = np.array(next_states)
                distance = np.linalg.norm(expert_traj[i*batch_size:(i+1)*batch_size, num_inputs:] - next_states, ord=2, axis=1)
                all_distance.append(distance)
    all_distance = np.concatenate(all_distance, axis=0)
    #feasibility = 1.- (all_distance-np.min(all_distance))/(np.max(all_distance)-np.min(all_distance))
    feasibility = np.exp(-all_distance/3.)
    return feasibility

def compute_feasibility_traj(expert_trajs, model, f_env, full_demo):
    all_distance = []
    for index in range(len(expert_trajs)):
        all_distance.append([])
        expert_traj = expert_trajs[index]
        with torch.no_grad():
            f_env.reset()
            #f_env.set_observation(expert_traj[0])
            jointposition = np.concatenate((full_demo[index][0][:9],np.array([0.03,0.03])),axis=None)
            f_env.panda._reset_robot(jointposition)
            for j in range(expert_traj.shape[0]-1):
                action = model(torch.from_numpy(np.concatenate([expert_traj[j:j+1], expert_traj[j+1:j+2]], axis=1)).float().to(device)).cpu().numpy()
                next_state, _, _, _ = f_env.step(action)
                all_distance[-1].append(np.linalg.norm(expert_traj[j+1] - next_state['ee_position'], ord=2, axis=0))
        all_distance[-1] = np.mean(all_distance[-1])
    all_distance = np.array(all_distance)
    #feasibility = 1.- (all_distance-np.min(all_distance))/(np.max(all_distance)-np.min(all_distance))
    feasibility = np.exp(-all_distance/3.)
    return feasibility

inverse_model = InverseModel(num_inputs*2, args.hidden_dim, num_actions, 6).float()
inverse_optimizer = optim.Adam(inverse_model.parameters(), 0.01)

train_inverse_dynamics(args.feasible_traj_paths, inverse_model, num_inputs, num_actions)


#load full demo
full_demo1 = pickle.load(open('..\\logs\\data\\infeasible_traj_9_1_5dis_full.pkl', 'rb'))
full_demo2 = pickle.load(open('..\\logs\\data\\infeasible_traj_9_1_0528_full.pkl', 'rb'))
full_demo = np.concatenate((full_demo1,full_demo2),axis=0)
expert_pairs, expert_trajs, pair_traj_id = load_demos(args.demo_files)    
feasibility_traj = compute_feasibility_traj(expert_trajs, inverse_model, f_env, full_demo)
#pdb.set_trace()
#feasibility = feasibility_traj[pair_traj_id]
pickle.dump(feasibility_traj, open('..\\logs\\data\\baseline\\baselinefea1000_3.pkl', 'wb'))
'''#expert_traj = np.concatenate(expert_pairs, axis=0)
#pdb.set_trace()
expert_traj = np.array(expert_pairs)

policy_net = Policy(num_inputs, num_actions, args.hidden_dim)
value_net = Value(num_inputs, args.hidden_dim).to(device)
discriminator = Discriminator(num_inputs + num_inputs, args.hidden_dim).to(device)
disc_criterion = nn.BCEWithLogitsLoss()
value_criterion = nn.MSELoss()
disc_optimizer = optim.Adam(discriminator.parameters(), args.lr)
value_optimizer = optim.Adam(value_net.parameters(), args.vf_lr)

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def update_params(batch):
    rewards = torch.Tensor(batch.reward).to(device)
    masks = torch.Tensor(batch.mask).to(device)
    actions = torch.Tensor(np.concatenate(batch.action, 0)).to(device)
    states = torch.Tensor(batch.state).to(device)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1).to(device)
    deltas = torch.Tensor(actions.size(0),1).to(device)
    advantages = torch.Tensor(actions.size(0),1).to(device)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    batch_size = math.ceil(states.shape[0] / args.vf_iters)
    idx = np.random.permutation(states.shape[0])
    for i in range(args.vf_iters):
        smp_idx = idx[i * batch_size: (i + 1) * batch_size]
        smp_states = states[smp_idx, :]
        smp_targets = targets[smp_idx, :]
        
        value_optimizer.zero_grad()
        value_loss = value_criterion(value_net(Variable(smp_states)), smp_targets)
        value_loss.backward()
        value_optimizer.step()

    advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states.cpu()))
    fixed_log_prob = normal_log_density(Variable(actions.cpu()), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=None):
        action_means, action_log_stds, action_stds = policy_net(Variable(states.cpu()))
        log_prob = normal_log_density(Variable(actions.cpu()), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages.cpu()) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()


    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states.cpu()))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)

def expert_reward(states, actions):
    states = np.concatenate(states)
    actions = np.concatenate(actions)
    with torch.no_grad():
        state_action = torch.Tensor(np.concatenate([states, actions], 1)).to(device)
        return -F.logsigmoid(discriminator(state_action)).cpu().detach().numpy()


def evaluate(episode, best_reward, log_file):
  with torch.no_grad():
    avg_reward = 0.0
    env.seed(1234)
    for _ in range(args.eval_epochs):
        state = env.reset()
        for _ in range(10000): # Don't infinite loop while learning
            state = torch.from_numpy(state['ee_position']).unsqueeze(0)
            action, _, _ = policy_net(Variable(state))
            action = action.data[0].numpy()
            next_state, reward, done, _ = env.step(action)
            avg_reward += reward
            if done:
                break
            state = next_state
    print('Evaluation: Episode ', episode, ' Reward ', avg_reward / args.eval_epochs)
    log_file.write('Evaluation: Episode '+str(episode)+' Reward '+str(avg_reward / args.eval_epochs)+'\n')
    log_file.flush()
    if best_reward < avg_reward / args.eval_epochs:
        best_reward = avg_reward / args.eval_epochs
        torch.save({'policy':policy_net.state_dict(), 'value':value_net.state_dict(), 'rew':best_reward}, save_path)

all_idx = np.arange(0, expert_traj.shape[0])
p_idx = np.random.permutation(expert_traj.shape[0])
expert_traj = expert_traj[p_idx][:]
feasibility = feasibility[p_idx]

feasibility = feasibility / (np.sum(feasibility)+0.0000001)
feasibility[0] = 1-np.sum(feasibility[1:])

best_reward = -1000000

for i_episode in range(args.num_epochs):
    env.seed(int(time.time()))
    memory = Memory()

    num_steps = 0
    num_episodes = 0
    
    reward_batch = []
    states = []
    actions = []
    next_states = []
    mem_actions = []
    mem_mask = []
    mem_next = []

    while num_steps < args.batch_size:
        state = env.reset()
   

        reward_sum = 0
        for t in range(10000): # Don't infinite loop while learning
            action = select_action(state['ee_position'])
            action = action.data[0].numpy()
            states.append(np.array([state['ee_position']]))
            actions.append(np.array([action]))
            next_state, true_reward, done, _ = env.step(action)
            next_states.append(np.array([next_state['ee_position']]))
            reward_sum += true_reward

            mask = 1
            if done:
                mask = 0

            mem_mask.append(mask)
            mem_next.append(next_state)

            if done:
                break

            state = next_state
        num_steps += (t-1)
        num_episodes += 1

        reward_batch.append(reward_sum)

    if i_episode % args.eval_interval == 0:
        evaluate(i_episode, best_reward, log_file)

    rewards = expert_reward(states, next_states)
    for idx in range(len(states)):
        memory.push(states[idx][0], actions[idx], mem_mask[idx], mem_next[idx], \
                    rewards[idx][0])
    batch = memory.sample()
    update_params(batch)

    ### update discriminator ###
    next_states = torch.from_numpy(np.concatenate(next_states))
    states = torch.from_numpy(np.concatenate(states))
   

    labeled_num = min(expert_traj.shape[0], num_steps)

    idx = np.random.choice(all_idx, labeled_num, p=feasibility.reshape(-1))

    expert_state_action = expert_traj[idx][:]
    pdb.set_trace()
    expert_state_action = torch.Tensor(expert_state_action).to(device)
    real = discriminator(expert_state_action)    

    state_action = torch.cat((states, next_states), 1).to(device)
    fake = discriminator(state_action)

    disc_optimizer.zero_grad()
    disc_loss = disc_criterion(fake, torch.ones(fake.size(0), 1).to(device)) + \
                disc_criterion(real, torch.zeros(real.size(0), 1).to(device))
    
    
    disc_loss.backward()
    disc_optimizer.step()
    ############################

    if i_episode % args.log_interval == 0:
        print('Episode {}\tAverage reward: {:.2f}\tMax reward: {:.2f}\tLoss (disc): {:.2f}'.format(i_episode, np.mean(reward_batch), max(reward_batch), disc_loss.item()))
        log_file.write('Episode {}\tAverage reward: {:.2f}\tMax reward: {:.2f}\tLoss (disc): {:.2f}\n'.format(i_episode, np.mean(reward_batch), max(reward_batch), disc_loss.item()))
        log_file.flush()'''