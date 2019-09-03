from facewarehouse_generate_npy import load_personalized_blendshape
import numpy as np
import os

result_path = '/home/jingwang/Data/data/FaceWarehouse/preprocessed/'

n_people = 150
n_test = 0
if __name__ == '__main__':
    ten_train = []
    ten_test = []
    for i in range(1, n_people+1):
        tmp = load_personalized_blendshape(i)
        if i <= n_test:
            ten_test.append(tmp[0])
        else:
            ten_train.append(tmp[0])
    ten_train = np.array(ten_train, dtype=np.float32)
    ten_test = np.array(ten_test, dtype=np.float32)
    print(ten_train.shape, ten_test.shape)
    np.save(os.path.join(result_path, 'neutral_train'), ten_train)
    np.save(os.path.join(result_path, 'neutral_test'), ten_test)
