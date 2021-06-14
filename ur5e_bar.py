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
    data_num = 10.
    inter_value = np.square(np.std(data1))/data_num + np.square(np.std(data2))/data_num
    test_statistic = (np.mean(data1) - np.mean(data2))/np.sqrt(inter_value)
    df = np.square(inter_value) / (np.square(np.square(np.std(data1))/data_num)/(data_num-1) + np.square(np.square(np.std(data2))/data_num)/(data_num-1))
    q = 0.05
    test_statistic = abs(test_statistic)
    p = 1 - scipy.stats.t.cdf(test_statistic,df=df)
    return (test_statistic, scipy.stats.t.ppf(1-q, df), p, df)

font = {'size'   : 40}

matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams['font.serif']
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#matplotlib.rcParams['font.serif'] = 'Times'


colors = ['#005A32', '#B2182B', '#41AB5D', '#2166AC']
labels = [ '2IWIL', 'AIRL', 'CAIL', 'D-REX','GAIL', 'IC-GAIL', 'T-REX','SSRR']
method_color = [ colors[3], colors[6], colors[1], colors[0], colors[5], colors[4], colors[2],  colors[7]]
policy_reward = []
for file_ in ['ur5e_reward.csv']:
    single_reward = [[], [], [], [], [], [], [], []]
    with open(file_) as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i != 0:
                for j in range(len(line)):
                    single_reward[j].append(float(line[j]))
    policy_reward.append(single_reward)

print('statistical sig airl: ', compute_p_value(np.array(policy_reward[0][2]), np.array(policy_reward[0][1])))
print('statistical sig gail: ', compute_p_value(np.array(policy_reward[0][2]), np.array(policy_reward[0][4])))
print('statistical sig twoiwil: ', compute_p_value(np.array(policy_reward[0][2]), np.array(policy_reward[0][0])))
print('statistical sig trex: ', compute_p_value(np.array(policy_reward[0][2]), np.array(policy_reward[0][6])))
print('statistical sig icgail: ', compute_p_value(np.array(policy_reward[0][2]), np.array(policy_reward[0][5])))
print('statistical sig drex: ', compute_p_value(np.array(policy_reward[0][2]), np.array(policy_reward[0][3])))
print('statistical sig ssrr: ', compute_p_value(np.array(policy_reward[0][2]), np.array(policy_reward[0][7])))

print('mean variance cail: ', np.array(policy_reward[0][2]).mean(axis=0), np.array(policy_reward[0][2]).std(axis=0))
print('mean variance airl: ', np.array(policy_reward[0][1]).mean(axis=0), np.array(policy_reward[0][1]).std(axis=0))
print('mean variance gail: ', np.array(policy_reward[0][4]).mean(axis=0), np.array(policy_reward[0][4]).std(axis=0))
print('mean variance twoiwil: ', np.array(policy_reward[0][0]).mean(axis=0), np.array(policy_reward[0][0]).std(axis=0))
print('mean variance trex: ', np.array(policy_reward[0][6]).mean(axis=0), np.array(policy_reward[0][6]).std(axis=0))
print('mean variance icgail: ', np.array(policy_reward[0][5]).mean(axis=0), np.array(policy_reward[0][5]).std(axis=0))
print('mean variance drex: ', np.array(policy_reward[0][3]).mean(axis=0), np.array(policy_reward[0][3]).std(axis=0))
print('mean variance ssrr: ', np.array(policy_reward[0][7]).mean(axis=0), np.array(policy_reward[0][7]).std(axis=0))

mean_reward = [[], [], [], []]
std_reward = [[], [], [], []]
for i in range(len(policy_reward)):
    for j in range(len(policy_reward[i])):
        mean_reward[j].append(np.mean(policy_reward[i][j]))
        std_reward[j].append(np.std(policy_reward[i][j]))
x = []
half_bar_width = 1.2
for i in range(len(method_color)):
    x.append(i*2+1)

orders = [6,4,7,2,0,5,3,1]
for i in range(1):
    fig, ax = plt.subplots(figsize=(10,10))
    data = [mean_reward[j][i] for j in range(len(method_color)) if len(mean_reward[j]) > 0]
    std_data = [std_reward[j][i] for j in range(len(method_color)) if len(mean_reward[j]) > 0]
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
    plt.savefig("ur5e.pdf", bbox_inches='tight', pad_inches=0)
