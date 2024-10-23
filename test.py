import os
import numpy as np

dataset_path = 'data/experiment_data/replica'
scans = os.listdir(dataset_path)
print(scans)

for i, name in enumerate(scans):
    print(i, name)


cond_idx = np.arange(0, 5, 1)
print(cond_idx)



'''
['scan65', 'scan115', 'scan210', 'scan710', 'scan310', 'scan55', 'scan215', 'scan610', 'scan815', 'scan510', 'scan75', 'scan715', 'scan615', 'scan25', 'scan410', 'scan110', 'scan15', 'scan315', 'scan45', 'scan415', 'scan810', 'scan85', 'scan35', 'scan515']
'''