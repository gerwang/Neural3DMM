import sys

template = sys.executable + ' neural3DMM.py --nz 50 --name paper_arch_vae_{0} --dataset FW_true_10000 --kl ' \
                            '{0} > /dev/null 2>&1 &'

with open('tasks.txt', 'w') as fout:
    for i in range(5, 10):
        for j in range(1, 10, 3):
            kl = j * 10 ** -i
            cmd = template.format(kl)
            fout.write(cmd + '\n')
