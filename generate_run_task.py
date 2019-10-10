import sys

template = sys.executable + ' neural3DMM.py --nz 50 --name pca_init_{0} --dataset FW_true_10000 --amount 3640 ' \
                            '--n_inter {0} > /dev/null 2>&1 & '

with open('tasks.txt', 'w') as fout:
    for num in [50, 51, 55, 100, 150, 250, 300]:
        fout.write(template.format(num) + '\n')
