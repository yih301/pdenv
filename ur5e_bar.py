import matplotlib.pyplot as plt 
import matplotlib
import pickle
import numpy as np
import pdb

import sys
import argparse

import os
import csv

import scipy.stats

def compute_p_value(data1, data2):
    data_num = 100.
    inter_value = np.square(np.std(data1))/data_num + np.square(np.std(data2))/data_num
    test_statistic = (np.mean(data1) - np.mean(data2))/np.sqrt(inter_value)
    df = np.square(inter_value) / (np.square(np.square(np.std(data1))/data_num)/(data_num-1) + np.square(np.square(np.std(data2))/data_num)/(data_num-1))
    q = 0.05
    test_statistic = abs(test_statistic)
    p = 1 - scipy.stats.t.cdf(test_statistic,df=df)
    return (test_statistic, scipy.stats.t.ppf(1-q, df), p, df)

font = {'size'   : 80}

matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams['font.serif']
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#matplotlib.rcParams['font.serif'] = 'Times'


colors = ['#2C2D2E', '#A8A4CE', '#3770AD', '#F99D59']  #color?
labels = [ 'GAIL','SAIL','ID-Feas','Ours'] #4321
method_color = [ colors[0], colors[1], colors[2], colors[3]]
policy_reward = []
policy_success = []
#change to pkl file loader
file_list = [
'..\\logs\\plotdata\\gail_reward.pkl',
'..\\logs\\plotdata\\sail_reward.pkl',
'..\\logs\\plotdata\\id_reward.pkl',
'..\\logs\\plotdata\\fea_reward.pkl']
file_list2 = [
'..\\logs\\plotdata\\gail_success.pkl',
'..\\logs\\plotdata\\sail_success.pkl',
'..\\logs\\plotdata\\id_success.pkl',
'..\\logs\\plotdata\\fea_success.pkl',]
for i in range(len(file_list)):
    policy_reward.append(pickle.load(open(file_list[i], 'rb')))
success = []
for i in range(len(file_list2)):
    policy_success.append(pickle.load(open(file_list2[i], 'rb')))

#p value for reward
print('statistical sig gail reward: ', compute_p_value(np.array(policy_reward[3]), np.array(policy_reward[0]))[2])
print('statistical sig sail reward: ', compute_p_value(np.array(policy_reward[3]), np.array(policy_reward[1]))[2])
print('statistical sig ID-Feas reward: ', compute_p_value(np.array(policy_reward[3]), np.array(policy_reward[2]))[2])

#p value for success rate
print('statistical sig gail success: ', compute_p_value(np.array(policy_success[3]), np.array(policy_success[0]))[2])
print('statistical sig sail success: ', compute_p_value(np.array(policy_success[3]), np.array(policy_success[1]))[2])
print('statistical sig ID-Feas success: ', compute_p_value(np.array(policy_success[3]), np.array(policy_success[2]))[2])


print('mean variance ours reward: ', np.array(policy_reward[0]).mean(axis=0), np.array(policy_reward[0]).std(axis=0))
print('mean variance gail reward: ', np.array(policy_reward[1]).mean(axis=0), np.array(policy_reward[1]).std(axis=0))
print('mean variance ID-Feas reward: ', np.array(policy_reward[2]).mean(axis=0), np.array(policy_reward[2]).std(axis=0))
print('mean variance sail reward: ', np.array(policy_reward[3]).mean(axis=0), np.array(policy_reward[3]).std(axis=0))

print('mean variance ours success: ', np.array(policy_success[0]).mean(axis=0), np.array(policy_success[0]).std(axis=0))
print('mean variance gail success: ', np.array(policy_success[1]).mean(axis=0), np.array(policy_success[1]).std(axis=0))
print('mean variance ID-Feas success: ', np.array(policy_success[2]).mean(axis=0), np.array(policy_success[2]).std(axis=0))
print('mean variance sail success: ', np.array(policy_success[3]).mean(axis=0), np.array(policy_success[3]).std(axis=0))

mean_reward = []
std_reward = []
for i in range(len(policy_reward)):
    mean_reward.append(np.mean(policy_reward[i]))
    std_reward.append(np.std(policy_reward[i]))


mean_success = []
std_success = []
for i in range(len(policy_success)):
    mean_success.append(np.mean(policy_success[i]))
    std_success.append(np.std(policy_success[i]))

x = []
half_bar_width = 1.2
for i in range(len(method_color)):
    x.append(i*2+1)

orders = [0,1,2,3]  #order?
for i in range(1):
    fig, ax = plt.subplots(figsize=(15,15))
    data = [mean_reward[j] for j in range(len(method_color)) if len(mean_reward) > 0]
    std_data = [std_reward[j] for j in range(len(method_color)) if len(mean_reward) > 0]
    for j in range(len(data)):
        ax.bar(x[0:len(data)][orders[j]], data[j], half_bar_width, color=method_color[j], label=labels[j])  
    ax.errorbar(np.array(x[0:len(data)])[orders], data, yerr=std_data, color='#000000', elinewidth=2, linewidth=0, capsize=6, capthick=2)
    ax.axhline(y=0, xmin=-2, xmax=x[-1]+2, color='#000000',linewidth=1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    #legend = plt.legend(loc='lower center', shadow=True, fontsize='x-large')
    #ax.set_xticks('off')
    #ax.set_xticklabels(labels)
    #legend = plt.legend(loc='lower center', shadow=True)
    plt.xticks([])
    plt.xlim(-2, x[-1]+2)
    #plt.xlabel('Method')
    plt.ylabel('Expected Return')

    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig("simulation_barplot_reward.pdf", bbox_inches='tight', pad_inches=0)

for i in range(1):
    fig, ax = plt.subplots(figsize=(15,15))
    data = [mean_success[j] for j in range(len(method_color)) if len(mean_success) > 0]
    std_data = [std_success[j] for j in range(len(method_color)) if len(mean_success) > 0]
    for j in range(len(data)):
        ax.bar(x[0:len(data)][orders[j]], data[j], half_bar_width, color=method_color[j], label=labels[j])  
    ax.errorbar(np.array(x[0:len(data)])[orders], data, yerr=std_data, color='#000000', elinewidth=2, linewidth=0, capsize=6, capthick=2)
    ax.axhline(y=0, xmin=-2, xmax=x[-1]+2, color='#000000',linewidth=1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    #legend = plt.legend(loc='lower center', shadow=True, fontsize='x-large')
    #ax.set_xticks('off')
    #ax.set_xticklabels(labels)
    #legend = plt.legend(loc='lower center', shadow=True)
    plt.xticks([])
    plt.xlim(-2, x[-1]+2)
    #plt.xlabel('Method')
    plt.ylabel('Success Rate')

    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig("simulation_barplot_success.pdf", bbox_inches='tight', pad_inches=0)
