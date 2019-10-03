import os

import numpy as np

input_paths = [
    '/home/gerw/Documents/git-task/coma/data/FW_fusion_10000/mean_data.npy',
    '/run/media/gerw/HDD/data/CoMA/data/FW_same/train.npy',
    '/run/media/gerw/HDD/data/CoMA/data/FW_aligned_10000/train.npy'
]

output_dir = '/run/media/gerw/HDD/data/CoMA/data/FW_fusion_10000'

np_output_path = os.path.join(output_dir, 'train.npy')
tag_output_path = os.path.join(output_dir, 'train_tag.npy')

inputs = [np.load(path) for path in input_paths]

tag = np.zeros((sum(x.shape[0] for x in inputs), 2), np.int16)
output = np.concatenate(inputs, axis=0)

# train 0 is tagged

n_people = 140
n_exp = 47

assert n_people * n_exp == inputs[1].shape[0]

cnt = 0

# mean data

for j in range(n_exp):
    tag[cnt] = 0, j
    cnt += 1  # prevbug: ftl

# labeled data

for i in range(n_people):
    for j in range(n_exp):
        tag[cnt] = i + 1, j
        cnt += 1

# unlabeled data

while cnt < output.shape[0]:
    tag[cnt] = -1, 0  # neutral expression
    cnt += 1

np.save(np_output_path, output)
np.save(tag_output_path, tag)
