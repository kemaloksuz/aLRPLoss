import json
import matplotlib.pyplot as plt
import numpy as np
import pdb
# read losses
loss_dict = [json.loads(line)for line in open('/home/cancam/workspace/aLRP-Loss/models/retinanet_APLoss/loss_log.json', 'r')]

# append to lists
cls_loss=[]
reg_loss=[]

length=len(loss_dict)
for i in loss_dict:
	cls_loss.append(i['cls_loss_current'])
	reg_loss.append(i['regression_loss_current'])

print(len(cls_loss),len(reg_loss))

# plot losses.
plt.plot(np.asarray(cls_loss))
plt.plot(np.asarray(reg_loss))
plt.plot(np.asarray(cls_loss)+np.asarray(reg_loss))
plt.legend(['aLRP Cls Comp. (AP)','aLRP Reg Comp.','aLRP Loss'])

plt.ylabel('Loss Values')
plt.xlabel('Iteration/50')

plt.show()
