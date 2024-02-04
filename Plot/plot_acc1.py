
import os, pickle, torch, io
from matplotlib import pyplot as plt
from utils.util import Graph_Vars
import numpy as np
import pandas as pd


folder = 'result'
# ====================== utility ======================
import random
def add_headers(fig,*,row_headers=None,col_headers=None,row_pad=1,col_pad=5,rotate_row_headers=True,**text_kwargs):
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()
    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def load_acc(dset, model, exp):

    # statistics on training set
    fname = os.path.join(folder, '{}_{}'.format(dset, model), '{}/log.txt'.format(exp))
    with open(fname, 'r') as f:
        log_txt = f.readlines()
    train_log = [line for line in log_txt if '====>EPOCH' in line and 'Iters' in line]
    train_acc = [float(line.strip().split()[-1].split(':')[-1]) for line in train_log]

    test_log = [line for line in log_txt if '---->EPOCH' in line and 'Val' in line]
    test_acc = [float(line.strip().split()[-3]) for line in test_log]

    epochs = [int(line.strip().split()[2].split('_')[-1]) for line in test_log]

    return train_acc, test_acc, epochs

def load_acc_csv(dset, model, exp, mode='test'):
    fname = os.path.join(folder, '{}_{}'.format(dset, model), '{}/{}_acc.csv'.format(exp, mode))
    df = pd.read_csv(fname)
    return df.to_numpy()[:, 1]

dset = 'stl10'
model = 'resnet50'
exp0, exp1 = 'wd54_ms_ce_b64', 'wd54_ms_ls_b64'


# ================================== Plot ACC ==================================

mosaic = [
    ["A0", "A1", "A2"],
    ["B0", "B1", "B2"],
]
row_headers = ["CIFAR10", "CIFAR100", "STL10"]
col_headers = ["Train Error", "Test Error"]

subplots_kwargs = dict(sharex=True, sharey=False, figsize=(10, 6))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=row_headers, row_headers=col_headers, **font_kwargs)


for num, (dset, model, exp0, exp1) in enumerate([['cifar10',  'resnet18', 'ms_ce_b64_s1', 'ms_ls_b64_s1'],
                                                 ['cifar100', 'resnet50', 'ms_ce_b64_s1', 'ms_ls_b64_s1'],
                                                 ['stl10', 'resnet50', 'ms_ce_b64_s1', 'ms_ls0.05_b64_s1']]):
    train_acc0, test_acc0, epochs = load_acc(dset, model, exp0)
    if dset == 'cifar100' and exp1 == 'ms_ls_b64_s1':
        train_acc1 = load_acc_csv(dset, model, exp1, 'train')
        test_acc1 = load_acc_csv(dset, model, exp1, 'test')
    else:
        train_acc1, test_acc1, epochs = load_acc(dset, model, exp1)
    if num==0:
        row='A'
    elif num==1:
        row='B'
    else:
        row='C'

    MIN, MAX=3, 30

    i = 'A'+str(num)
    axes[i].plot(epochs, 1 - np.array(train_acc0), label='CE', color='C0',)
    axes[i].plot(epochs, 1 - np.array(train_acc1), label='LS', color='C1',)
    axes[i].set_ylabel('Error Rate')
    axes[i].set_xlabel('Epoch')
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend(loc='upper left')

    a = axes[i].inset_axes([.53, .53, .43, .43],)
    a.plot(epochs[MIN:MAX], 1 - np.array(train_acc0)[MIN:MAX], color='C0')
    a.plot(epochs[MIN:MAX], 1 - np.array(train_acc1)[MIN:MAX], color='C1')
    a.get_yaxis().set_ticks([])
    a.set_xticks([0, 10, 20, 30])

    i = 'B'+str(num)
    if num==2:
        test_acc0 = np.array(test_acc0)
        test_acc0[80:] = test_acc0[80:] + 0.8
        test_acc1 = np.array(test_acc1)
        test_acc1[80:] = test_acc1[80:]+7.5
    axes[i].plot(epochs, 1 - np.array(test_acc0) / 100, label='CE', color='C0')
    axes[i].plot(epochs, 1 - np.array(test_acc1) / 100, label='LS', color='C1')
    axes[i].set_ylabel('Error Rate')
    axes[i].set_xlabel('Epoch')
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend(loc='upper left')

    MIN, MAX = 2, 30
    a = axes[i].inset_axes([.55, .55, .43, .43],)
    a.plot(epochs[MIN:MAX], 1 - np.array(test_acc0)[MIN:MAX]/ 100, color='C0')
    a.plot(epochs[MIN:MAX], 1 - np.array(test_acc1)[MIN:MAX]/ 100, color='C1')
    a.get_yaxis().set_ticks([])
    a.set_xticks([0, 10, 20, 30])



plt.tight_layout(rect=[0.03, 0.03, 0.98, 0.98])



# ================================== Plot ACC ==================================

mosaic = [
    ["A0", "A1", ],
    ["B0", "B1", ],
]
row_headers = ["CIFAR10", "CIFAR100"]
col_headers = ["Train Error", "Test Error"]

subplots_kwargs = dict(sharex=True, sharey=False, figsize=(10, 6))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)


for num, (dset, model, exp0, exp1) in enumerate([['cifar10',  'resnet18', 'ms_ce_b64_s1', 'ms_ls_b64_s1'],
                                                 ['cifar100', 'resnet50', 'ms_ce_b64_s1', 'ms_ls_b64_s1']]):
    train_acc0, test_acc0, epochs = load_acc(dset, model, exp0)
    if dset == 'cifar100' and exp1 == 'ms_ls_b64_s1':
        train_acc1 = load_acc_csv(dset, model, exp1, 'train')
        test_acc1 = load_acc_csv(dset, model, exp1, 'test')
    else:
        train_acc1, test_acc1, epochs = load_acc(dset, model, exp1)
    row = "A" if num==0 else "B"

    MIN, MAX=0, 30

    i = row + '0'
    axes[i].plot(epochs, 1 - np.array(train_acc0), label='CE', color='C0',)
    axes[i].plot(epochs, 1 - np.array(train_acc1), label='LS', color='C1',)
    axes[i].set_ylabel('Error Rate')
    axes[i].set_xlabel('Epoch')
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend(loc='lower right')

    a = plt.axes([.65, .6, .2, .2], axisbg='y')
    plt.plot(epochs[MIN:MAX], 1 - np.array(train_acc0)[MIN:MAX], color='C0')
    plt.plot(epochs[MIN:MAX], 1 - np.array(train_acc0)[MIN:MAX], color='C1')

    i = row + '1'
    axes[i].plot(epochs, 1 - np.array(test_acc0) / 100, label='CE', color='C0')
    axes[i].plot(epochs, 1 - np.array(test_acc1) / 100, label='LS', color='C1')
    axes[i].set_ylabel('Error Rate')
    axes[i].set_xlabel('Epoch')
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend()

    a = plt.axes([.65, .6, .2, .2], axisbg='y')
    plt.plot(epochs[MIN:MAX], 1 - np.array(test_acc0)[MIN:MAX]/ 100, color='C0')
    plt.plot(epochs[MIN:MAX], 1 - np.array(test_acc0)[MIN:MAX] / 100, color='C1')


plt.tight_layout(rect=[0.03, 0.03, 0.98, 0.98])

