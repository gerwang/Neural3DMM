import sys

template = sys.executable + ' neural3DMM.py --nz 50 --name paper_arch --dataset FW_true_10000 --amount {} > /dev/null ' \
                            '2>&1 & '

with open('tasks.txt', 'w') as fout:
    for num in list(range(140, 10 ** 4 + 1, 500)) + [10 ** 4]:
        fout.write(template.format(num) + '\n')
