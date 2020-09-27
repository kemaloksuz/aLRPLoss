import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pdb

def load_json_logs(json_logs):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics, e.g. memory, bbox_mAP
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        counter = 0
        with open(json_log, 'r') as log_file:
            for l in log_file:
                log = json.loads(l.strip())
                if counter not in log_dict:
                    log_dict[counter] = defaultdict(list)
                for k, v in log.items():
                    log_dict[counter][k].append(v)
        counter+=1
    return log_dicts


def main():
    json_logs = ['./speed_logs/APLoss500_r50_TRUBA.json']    
    for json_log in json_logs:
        assert json_log.endswith('.json')
    log_dicts = load_json_logs(json_logs)
    #iters_per_epoch = int(np.array(log_dicts[num_run[i]][0]['fg_grad_magn']).shape[0]/12)
    speed_per_iteration = np.array(log_dicts[0][0]['time'])[1:].mean()
    speed_per_image = speed_per_iteration/32
    print("aLRP Loss iteration time=", speed_per_iteration)
    print("aLRP Loss iteration time=", speed_per_image)
    #print("AP Loss iteration time=", np.array(log_dicts[0][0]['time']).mean())

'''
    for i in range(4):
        print(i)
    	#loc_grad= np.array(log_dicts[i][0]['loc_grad_magn'])
        if average_over_epochs==0:
            rates = np.zeros([num_run[i+1],np.array(log_dicts[num_run[i]][0]['fg_grad_magn']).shape[0]])        
            for j in range(num_run[i+1]):
                rates[j]= (np.array(log_dicts[num_run[:i+1].sum()+j][0]['bg_grad_magn'])/(np.array(log_dicts[num_run[:i+1].sum()+j][0]['fg_grad_magn'])+1e-6))
            rate_mean = np.mean(rates, axis=0)
            if i==0 or i==1:
                rate_std = np.std(rates, axis=0)
        else:

            rate_mean = np.zeros([total_epochs*iters_per_epoch])
            rate_std = np.zeros([total_epochs*iters_per_epoch])

            epoch_mean = np.zeros([total_epochs])
            epoch_std = np.zeros([total_epochs])

            rates = np.zeros([num_run[i+1], total_epochs*iters_per_epoch])        

            for j in range(num_run[i+1]):
                rates[j] = (np.array(log_dicts[num_run[:i+1].sum()+j][0]['bg_grad_magn'])/(np.array(log_dicts[num_run[:i+1].sum()+j][0]['fg_grad_magn'])+1e-6))[:total_epochs*iters_per_epoch]
               
            if i==0 or i==1:
                for j in range(total_epochs):
                    epoch_mean[j] = np.mean(rates[:,j*iters_per_epoch:(j+1)*iters_per_epoch])
                    epoch_std[j] = np.std(rates[:,j*iters_per_epoch:(j+1)*iters_per_epoch])
                    rate_mean[j*iters_per_epoch:(j+1)*iters_per_epoch] = epoch_mean[j]
                    rate_std[j*iters_per_epoch:(j+1)*iters_per_epoch] = epoch_std[j]
            else:
                rate_mean = rates[0,:total_epochs*iters_per_epoch]
        size = rate_mean.shape[0]
        size_FL_CE = epoch_mean.shape[0]
        
        min_rate = 1./(np.nanmin(rates))
        max_rate = np.nanmax(rates)
#        print(i,np.unravel_index(rates.argmax(), rates.shape), rates.max())
        cell_text.append([' ','1/%1.3f' % min_rate, '%1.3f' %  max_rate])
#        ax.plot(np.arange(size)*50, rate_mean+rate_std, colors[i], linewidth = 0.5)
        

        if i==0:
            ax.errorbar(np.arange(size_FL_CE)*50*iters_per_epoch, epoch_mean, yerr=epoch_std,fmt='o', capsize=3, capthick=5, ecolor=colors[i], elinewidth=2, mfc=colors[i], mec=colors[i], mew=0, ms=4)
            ax.plot(np.arange(size_FL_CE)*50*iters_per_epoch, epoch_mean,color=colors[i], linewidth = 1.)

            ax.plot((95000+110000)/2, location[i], 'o', mfc=colors[i], mew=0, ms=6)

            ax.arrow(int(50*iters_per_epoch/3)+6500, 2.08, 0, 0.1, color= colors[i], width=0.2, head_length=0.05, head_width=2500)
            ax.text(10000, 2.15,r"$\mu = %1.1f, \sigma = %1.1f$" %  (epoch_mean[0], epoch_std[0]),  color= colors[i], fontsize=8 )
            #ax.fill_between(np.arange(size)*50, rate_mean-rate_std, rate_mean+rate_std, facecolor=colors[i], alpha=0.25)
            #ax.plot(np.arange(size)*50, rate_mean, colors[i], linewidth = 1.5)
        elif i==1:
            ax.errorbar(np.arange(size_FL_CE)*50*iters_per_epoch+int(50*iters_per_epoch/3), epoch_mean, yerr=epoch_std, fmt='^', capsize=3, capthick=5, ecolor=colors[i], elinewidth=2, mfc=colors[i], mec=colors[i], mew=0, ms=4)  
            ax.plot((95000+110000)/2, location[i], '^', mfc=colors[i], mew=0, ms=6)
            ax.plot(np.arange(size_FL_CE)*50*iters_per_epoch, epoch_mean,color=colors[i], linewidth = 1.)

            #ax.fill_between(np.arange(size)*50, rate_mean-rate_std, rate_mean+rate_std, facecolor=colors[i], alpha=0.25)
            #ax.plot(np.arange(size)*50, rate_mean, colors[i], linewidth = 1.5)
        elif i==2:
            loc=73500
            ax.plot(np.arange(size)*50, rate_mean, colors[i], linewidth = 1.)
            #ax.plot([loc, loc], [1.0, 1.45],'--' ,color=colors[i], linewidth = 0.75)
            ax.text(loc, 1.45,"diverges",  color= colors[i], fontsize=8 )
            ax.arrow(loc, 1.0, 0, 0.4, color= colors[i], width=0.2, head_length=0.05, head_width=2500)

        else:
            ax.plot(np.arange(size)*50, rate_mean, colors[i], linewidth = 1.)
            
            #

        ax.plot([95000, 110000], [location[i], location[i]], colors[i], linewidth = 1.)
    
    matplotlib.rcParams.update({'font.size': fontsize})
    ax.set_xlabel('iterations', fontsize=fontsize)
    ax.set_ylabel(r"Magnitude Rate = $\left|\sum_{i\in\mathcal{N}}\frac{\partial\mathcal{L}}{\partial s_i}\right|  / \left| \sum_{i\in\mathcal{P}} \frac{\partial \mathcal{L}}{ \partial s_i}\right|$", fontsize=10)    

    rows = ('Cross Entropy', 'Focal Loss', r'$\mathcal{L}^{\mathrm{aLRP}}_{cls}$ with ${L^\mathrm{aLRP}_{ij}}^* = 0$', r'$\mathcal{L}^{\mathrm{aLRP}}_{cls}$')
    columns = ('Legend', 'Min Rate', 'Max Rate')
    the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      colLabels=columns,
                      bbox=[0.5, 0.6, 0.5, 0.4], 
                      cellLoc='center')
    the_table.auto_set_font_size(False)
    ax.set_ylim([0.5, 2.25])
    ax.set_xticks([0,25000,50000,75000,100000,125000,150000,175000])
    ax.set_xticklabels(['0K','25K','50K','75K','100K','125K','150K','175K'])
	
    plt.savefig('GradientComparison.pdf', edgecolor='none',format=''pdf',bbox_inches = 'tight')
'''

if __name__ == '__main__':
    main()
