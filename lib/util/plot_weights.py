import numpy as np
import json
import pdb
import os
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba

json_log='/home/cancam/imgworkspace/aLRP-Loss/models/coco/aLRP_r50_IoU_100e/aLRP_r50_IoU_100e-040640-20200503.json'
exp_name = os.path.split(json_log)[-1][:-5]
save_name = '/home/cancam/imgworkspace/aLRP-Loss/loss_plots/' + exp_name + '.pdf'

# keys = ['cls_loss', 'regression_loss', 'total', 'times', 'regression_weight']
# colors = ['r', 'g', 'b', 'c', 'k']
keys = ['regression_loss', 'total', 'cls_loss', 'regression_weight', 'times']
colors = ['g', 'b', 'r', 'k', 'c']
labels = [r'$\mathrm{aLRP_{reg}}$', r'$\mathrm{aLRP}$', r'$\mathrm{aLRP_{cls}}$',\
           r'$\mathrm{w_{aLRP_{reg}}}$', \
           r'$\mathrm{aLRP_{reg}} \times \mathrm{w_{aLRP_{reg}}}$']
keys.reverse()
colors.reverse()
labels.reverse()

#labels = [r'$\mathrm{aLRP_{cls}}$', r'$\mathrm{aLRP_{reg}}$', \
#          r'$\mathrm{aLRP}$', r'$\mathrm{aLRP_{reg}} \times \mathrm{w_{aLRP_{reg}}}$', \
#          r'$\mathrm{w_{aLRP_{reg}}}$']

def load_json_logs(json_log, keys):
    log_dict = dict()
    with open(json_log, 'r') as log_file:
        for l in log_file:
            log = json.loads(l.strip())
            if 'epoch' not in log:
                continue
            epoch = log.pop('epoch')
            if epoch not in log_dict:
                log_dict[epoch] = defaultdict(list)
            if 'total' in keys:
                log['total'] = log['cls_loss'] + log['regression_loss']
            if 'times' in keys:
                log['times'] = log['regression_loss'] * log['regression_weight']
            for k,v in log.items():
                log_dict[epoch][k].append(v)
    return log_dict

def plot_curve(log_dict, keys):
    metrics = keys
    epochs = list(log_dict.keys())
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for j, metric in enumerate(metrics):
        xs = []
        ys = []
        num_iters_per_epoch = log_dict[epochs[0]]['iter'][-1]
        for epoch in epochs:
            iters = log_dict[epoch]['iter']
            xs.append(np.array(iters) + (epoch)*num_iters_per_epoch)
            ys.append(np.array(log_dict[epoch][metric][:len(iters)]))
        
        xs = np.concatenate(xs)
        ys = np.concatenate(ys)
        if metric != 'regression_weight' and metric != 'times':
            ax1.plot(xs, ys, label=labels[j], color=colors[j], linewidth=4.0)
            ax1.set_xlabel('Iter')
            ax1.set_ylabel('Loss')
            ax1.set_ylim([0,1])
            ax1.set_xlim([0, xs.max()])
        elif metric == 'regression_weight':
            ax2.plot(xs, ys, label=labels[j], color=colors[j], linewidth=4.0)
            ax2.set_xlabel('Iter')
            ax2.set_ylabel('Regression Weight')
            ax2.set_ylim([0, ys.max()])
        elif metric == 'times':
            ax1.plot(xs, ys, label=labels[j], color=colors[j], linewidth=4.0)
            ax1.set_xlabel('Iter')
            #ax1.set_ylim([0, ys.max()])

    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    ax1.grid(alpha=0.75)
    plt.savefig(save_name,\
                bbox_inches='tight',\
                pad_inches=0.) 


if __name__ == '__main__':
    log_dict = load_json_logs(json_log, keys)
    plot_curve(log_dict, keys)
