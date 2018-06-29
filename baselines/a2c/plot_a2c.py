# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
import sys
import numpy as np
import json
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rc('text', usetex=True)
import matplotlib.patches as mpatches

color_list = sns.color_palette("muted")
sns.palplot(color_list)

seeds = range(1, 6)

def ema(x, alpha=0.95):
  res = []
  mu = 0.
  for val in x:
    mu = alpha*mu + (1 - alpha)*val
    res.append(mu)

  return np.array(res)

def plot(prefix, color, label):
  data = {}
  for seed in seeds:
    rewards = []
    steps = []
    with open(os.path.join('%s_%d' % (prefix, seed), 'monitor.json.monitor.csv'), 'Ur') as f:
      next(f)
      next(f)
      for line in f:
        reward, length, _ = line.strip().split(',')
        reward = float(reward)
        length = int(length)
        rewards.append(reward)
        steps.append(length)

      data[seed] = [steps, rewards]

    # Extract variances
    log_variances = []
    with open(os.path.join('%s_%d' % (prefix, seed), 'log.txt'), 'Ur') as f:
      for line in f:
        if 'log_variance' in line:
          log_variance = float(line.split()[3])
          log_variances.append(log_variance)
    data[seed].append(log_variances)

  # Get the mean curve for each plot
  smoother = lambda x: ema(x)

  plot_x = np.linspace(0, np.min([np.sum(steps)/1000. for steps, _, _ in data.values()]), 1000)
  rewards = [np.interp(plot_x, np.cumsum(steps)/1000., smoother(rewards))
             for steps, rewards, _ in data.values()]
  rewards_mean = np.mean(rewards, axis=0)
  rewards_std = np.std(rewards, axis=0)
  rewards_min = np.min(rewards, axis=0)
  rewards_max = np.max(rewards, axis=0)
  rewards_lo = np.clip(rewards_mean - rewards_std, rewards_min, rewards_max)
  rewards_hi = np.clip(rewards_mean + rewards_std, rewards_min, rewards_max)

  plt.subplot(1, 2, 1)
  plt.plot(plot_x, np.mean(rewards, axis=0), color=color, label=label if seed == 1 else None)
  plt.fill_between(plot_x, rewards_lo, rewards_hi, color=color, alpha=0.2)
  plt.xlabel('Steps (thousands)', fontsize=16)
  plt.ylabel('Average Reward', fontsize=16)
  plt.grid(alpha=0.5)
  plt.title('InvertedPendulum-v1', fontsize=18)

  min_len = np.min([len(log_variances) for _, _, log_variances in data.values()])
  plot_x = np.arange(min_len) * 10
  log_variances = [log_variances[:min_len] for _, _, log_variances in data.values()]
  log_variances_mean = np.mean(log_variances, axis=0)
  log_variances_std = np.std(log_variances, axis=0)
  log_variances_min = np.min(log_variances, axis=0)
  log_variances_max = np.max(log_variances, axis=0)
  log_variances_lo = np.clip(log_variances_mean - log_variances_std, log_variances_min, log_variances_max)
  log_variances_hi = np.clip(log_variances_mean + log_variances_std, log_variances_min, log_variances_max)

  plt.subplot(1, 2, 2)
  plt.plot(plot_x,
           log_variances_mean, color=color, label=label if seed == 1 else None)
  plt.fill_between(plot_x,
                   log_variances_lo, log_variances_hi, color=color, alpha=0.2)
  plt.xlabel('Iteration', fontsize=16)
  plt.ylabel('ln(Variance)', fontsize=16)
  plt.grid(alpha=0.5)
  plt.title('InvertedPendulum-v1', fontsize=18)


plt.figure(figsize=(20,6))

exps = [
    ('log_a2c', 'State-dependent baseline'),
    ('log_a2c_relax', 'LAX'),
]

for i, (prefix, label) in enumerate(exps):
  plot(prefix, color_list[i], label)

plt.legend(
    handles=[
        mpatches.Patch(color=color_list[i], label=label) for i, (_, label) in enumerate(exps)
    ],
    loc='upper center', bbox_to_anchor=(-0.10, -0.13),
    ncol=6,
           prop={'size': 14})

plt.savefig('InvertedPendulum-v1.pdf', bbox_inches='tight')


