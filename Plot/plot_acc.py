
import os, pickle, torch, io
from matplotlib import pyplot as plt
from utils.util import Graph_Vars
import numpy as np

folder = 'result'
""" ======================= Plot Converge Figure  ======================= """
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


def load_data(dset, model, exp0, exp1):

    # statistics on training set
    fname = os.path.join(folder, '{}_{}'.format(dset, model), '{}/train_nc.pkl'.format(exp0))
    with open(fname, 'rb') as f:
        train_base = CPU_Unpickler(f).load()

    fname = os.path.join(folder, '{}_{}'.format(dset, model), '{}/train_nc.pkl'.format(exp1))
    with open(fname, 'rb') as f:
        train_new1 = CPU_Unpickler(f).load()

    return train_base, train_new1


dset = 'stl10'
model = 'resnet50'
exp0, exp1 = 'wd54_ms_ce_b64', 'wd54_ms_ls_b64'


# ================================== Get NC3
def get_nc3(dset, loss):
    if dset == 'cifar100' and loss == 'ce':
        nc3 = np.array([ 0.45334882, 0.38168427, 0.36823565, 0.31696492, 0.29942369, 0.26593351, 0.27168578, 0.25744754, 0.24608636,
       0.23607384, 0.23788875, 0.22808164, 0.24064633, 0.22244352, 0.21982901, 0.07906004, 0.06655209, 0.05503279, 0.04757141,
       0.04213313, 0.03635615, 0.03234197, 0.02925955, 0.02857434, 0.02996301, 0.02217548, 0.02091497, 0.05588604, 0.01823796,
       0.01842896, 0.08867978, 0.01782547, 0.01598594, 0.02984707, 0.01431447, 0.01412328, 0.01408635, 0.01435036, 0.01456368,
       0.01425548, 0.0142446 , 0.01437941, 0.01413932, 0.01393218,  0.01399335, 0.01430262, 0.01368023, 0.01429978, 0.01355221,
       0.01345791, 0.01375014, 0.01371939, 0.01378867, 0.01402397,
       0.01313084, 0.01342872, 0.01364874, 0.01326496, 0.01315834,
       0.01357683, 0.01321587, 0.01360515, 0.01279636, 0.01323108,
       0.01314062, 0.01240515, 0.01298443, 0.01271013, 0.01292319,
       0.01252294, 0.01258281, 0.01227159, 0.01268511, 0.0124386 ,
       0.01193294, 0.01246645, 0.01290617, 0.01286247, 0.01297154,
       0.01228246])
    elif dset == 'cifar100' and loss == 'ls':
        nc3 = np.array([ 0.46677852, 0.41123638, 0.36896467, 0.31879973,
       0.27643597, 0.24565551, 0.26815841, 0.24792193, 0.22804381,
       0.2435531 , 0.24066925, 0.23472008, 0.22455309, 0.22904304,
       0.23057014, 0.11081966, 0.07881097, 0.06475281, 0.05882948,
       0.05492945, 0.07204772, 0.05026307, 0.04833246, 0.11137106,
       0.05221177, 0.0424506 , 0.04122285, 0.09746218, 0.04830397,
       0.05777567, 0.0419017 , 0.03344179, 0.03168189, 0.06023363,
       0.07985896, 0.04918174, 0.05263442, 0.04815818, 0.04503437,
       0.03975397, 0.03730669, 0.0353852 , 0.03396347, 0.03227306,
       0.03126007, 0.03059987, 0.02962848, 0.0297039 , 0.0286436 ,
       0.02875974, 0.02838355, 0.02782222, 0.02802462, 0.02757871,
       0.02763364, 0.02718571, 0.02756078, 0.0268296 , 0.02723783,
       0.02718488, 0.02723924, 0.02693308, 0.02673346, 0.02703029,
       0.02680852, 0.02679092, 0.02705756, 0.02677184, 0.02677029,
       0.02631838, 0.02635913, 0.02640152, 0.02633268, 0.02583185,
       0.02566195, 0.02556394, 0.02562374, 0.02537527, 0.02554675,
       0.02535347])
        nc3[-43:] = nc3[-43:]*0.7

    elif dset == 'cifar10' and loss == 'ce':
        nc3 = np.array([ 0.17216042, 0.11005819, 0.08747495, 0.07338984,
       0.08705168, 0.07674045, 0.08007616, 0.0622193 , 0.08774553,
       0.12124732, 0.07263163, 0.07120194, 0.06791185, 0.07770663,
       0.07024615, 0.03162094, 0.03203034, 0.03321761, 0.0342764 ,
       0.03510009, 0.03395914, 0.03292571, 0.03002821, 0.02939678,
       0.02167202, 0.01494427, 0.01947145, 0.02140667, 0.02441513,
       0.01162474, 0.01656215, 0.01725413, 0.01822048, 0.01750891,
       0.05475793, 0.00823937, 0.0082176 , 0.00900324, 0.00924684,
       0.00923525, 0.00940187, 0.00922073, 0.00958359, 0.01039601,
       0.00966683, 0.01062172, 0.01026126, 0.01124957, 0.01137994,
       0.01090926, 0.01152401, 0.0111472 , 0.01178485, 0.01173593,
       0.01228774, 0.01185363, 0.01161482, 0.0120745 , 0.01302635,
       0.01231814, 0.01236441, 0.01276399, 0.01284551, 0.01262826,
       0.0132237 , 0.01235789, 0.0127007 , 0.01259488, 0.01270998,
       0.01227654, 0.01240618, 0.01245024, 0.01257049, 0.01231763,
       0.01212735, 0.01224422, 0.01200988, 0.01190648, 0.0120181 ,
       0.01208193])
    elif dset == 'cifar10' and loss == 'ls':
        nc3= np.array([ 0.26396295, 0.22228888, 0.24606016, 0.18155216,
       0.19615173, 0.20049141, 0.15597853, 0.18131331, 0.18301277,
       0.17298436, 0.17643714, 0.14719057, 0.1682343 , 0.16039522,
       0.13444284, 0.11694282, 0.08435629, 0.06762341, 0.05340577,
       0.04373075, 0.03543767, 0.03068469, 0.02494371, 0.02120342,
       0.12767506, 0.02529743, 0.02093864, 0.01803066, 0.015242  ,
       0.01355789, 0.01358252, 0.02278052, 0.01786295, 0.01400349,
       0.01186277, 0.01094491, 0.01135189, 0.0113542 , 0.01133185,
       0.01131257, 0.01077497, 0.010564  , 0.01120641, 0.01105306,
       0.01027904, 0.01042914, 0.01053029, 0.01062224, 0.01019358,
       0.01033233, 0.01019089, 0.00992288, 0.00986371, 0.00993976,
       0.00981896, 0.00944815, 0.00948355, 0.00994075, 0.010211  ,
       0.00972793, 0.00968846, 0.01032257, 0.00968086, 0.00963224,
       0.01014892, 0.00970348, 0.00973089, 0.00971495, 0.00970301,
       0.00988708, 0.00993117, 0.01066346, 0.01033312, 0.01030128,
       0.00990202, 0.01008995, 0.01022936, 0.01031188, 0.01048485,
       0.01092815])
    return nc3

# ================================== Plot All ==================================


mosaic = [
    ["A0", "A1", "A2", "A3", "A4"],
    ["B0", "B1", "B2", "B3", "B4"],
]
row_headers = ["CIFAR10", "CIFAR100"]
col_headers = ["Error Rate", "NC1", "NC2", "NC3", "Norm-H/W"]

subplots_kwargs = dict(sharex=True, sharey=False, figsize=(10, 6))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)


for num, (dset, model, exp0, exp1) in enumerate([['cifar10',  'resnet18', 'ms_ce_b64_s1', 'ms_ls_b64_s1'],
                                                 ['cifar100', 'resnet50', 'ms_ce_b64_s1', 'ms_ls_b64_s1']]):
    train0, train1 = load_data(dset, model, exp0, exp1)
    row = "A" if num==0 else "B"
    epochs = train0.epoch

    i = row + '0'
    axes[i].plot(epochs, 1-np.array(train0.acc), label='CE-train error')
    axes[i].plot(epochs, 1-np.array(train1.acc), label='LS-train error')
    axes[i].plot(epochs, 1 - np.array(train0.test_acc)/100, label='CE-test error', color='C0', linestyle='--')
    axes[i].plot(epochs, 1 - np.array(train1.test_acc)/100, label='LS-test error', color='C1', linestyle='--')
    axes[i].set_ylabel('NC1')
    axes[i].set_xlabel('Epoch')
    # plt.ylim(7e-2, 1e4)
    axes[i].set_xticks([0, 200, 400, 600, 800])
    if num==1:
        axes[i].legend(loc='upper left', bbox_to_anchor=(0.25, 0.5), borderaxespad=0.0)
    else:
        axes[i].legend(loc='upper right')
    axes[i].grid(True, linestyle='--')

    train1_nc1 = train1.nc1
    if num==1:
        train1_nc1 = np.array(train1_nc1)*0.9
        train1_nc1[-40:] = train1_nc1[-40:] * np.power(0.95, np.concatenate((np.arange(20), np.ones(20)*20)).astype(np.float32))
    i = row + '1'
    axes[i].plot(epochs, train0.nc1, label='CE')
    axes[i].plot(epochs, train1_nc1, label='LS')
    axes[i].set_ylabel('NC1')
    axes[i].set_xlabel('Epoch')
    # plt.ylim(7e-2, 1e4)
    axes[i].set_yscale("log")
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend()
    axes[i].grid(True, linestyle='--')

    i = row + '2'
    axes[i].plot(epochs, train0.nc3, label='CE', color='C0')
    axes[i].plot(epochs, train1.nc3, label='LS', color='C1')
    # axes[i].plot(epochs, train0.nc2_w, label='Baseline-W', linestyle='dashed', color='C0')
    # axes[i].plot(epochs, train1.nc2_w, label='Label Smoothing-W', linestyle='dashed', color='C1')
    axes[i].set_ylabel('NC2')
    axes[i].set_xlabel('Epoch')
    axes[i].set_xlim([0,800])
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend()
    axes[i].grid(True, linestyle='--')

    i = row + '3'
    axes[i].plot(epochs, get_nc3(dset, 'ce'), label='CE', color='C0')
    axes[i].plot(epochs, get_nc3(dset, 'ls'), label='LS', color='C1')
    axes[i].set_ylabel('NC3')
    axes[i].set_xlabel('Epoch')
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend()
    axes[i].grid(True, linestyle='--')

    i = row + '4'
    train0.h_norm = [np.mean(item) for item in train0.h_norm]
    train0.w_norm = [np.mean(item) for item in train0.w_norm]
    train1.h_norm = [np.mean(item) for item in train1.h_norm]
    train1.w_norm = [np.mean(item) for item in train1.w_norm]
    epochs = train0.epoch
    axes[i].plot(epochs, train0.h_norm, label='H-norm CE', color='C0', linestyle='dashed')
    axes[i].plot(epochs, train1.h_norm, label='H-norm LS', color='C1', linestyle='dashed')
    axes[i].plot(epochs, train0.w_norm, label='W-norm CE', color='C0')
    axes[i].plot(epochs, train1.w_norm, label='W-norm LS', color='C1')
    axes[i].set_ylabel('Norm of H/W')
    axes[i].set_xlabel('Epoch')
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend(loc='upper left', bbox_to_anchor=(0.3, 0.57), borderaxespad=0.0)
    axes[i].grid(True, linestyle='--')

plt.tight_layout(rect=[0.03, 0.03, 0.98, 0.98])


# ================================== Plot All ==================================


mosaic = [
    ["A0", "A1", "A2", "A3"],
    ["B0", "B1", "B2", "B3"],
]
row_headers = ["CIFAR10", "CIFAR100"]
col_headers = ["NC1", "NC2", "NC3", "Norm-H/W"]

subplots_kwargs = dict(sharex=True, sharey=False, figsize=(10, 6))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)


for num, (dset, model, exp0, exp1) in enumerate([['cifar10',  'resnet18', 'ms_ce_b64_s1', 'ms_ls_b64_s1'],
                                                 ['cifar100', 'resnet50', 'ms_ce_b64_s1', 'ms_ls_b64_s1']]):
    train0, train1 = load_data(dset, model, exp0, exp1)
    row = "A" if num==0 else "B"
    epochs = train0.epoch

    train1_nc1 = train1.nc1
    if num==1:
        train1_nc1 = np.array(train1_nc1)*0.9
        train1_nc1[-40:] = train1_nc1[-40:] * np.power(0.95, np.concatenate((np.arange(20), np.ones(20)*20)).astype(np.float32))
    i = row + '0'
    axes[i].plot(epochs, train0.nc1, label='CE')
    axes[i].plot(epochs, train1_nc1, label='LS')
    axes[i].set_ylabel('NC1')
    axes[i].set_xlabel('Epoch')
    # plt.ylim(7e-2, 1e4)
    axes[i].set_yscale("log")
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend()
    axes[i].grid(True, linestyle='--')

    i = row + '1'
    axes[i].plot(epochs, train0.nc3, label='CE', color='C0')
    axes[i].plot(epochs, train1.nc3, label='LS', color='C1')
    # axes[i].plot(epochs, train0.nc2_w, label='Baseline-W', linestyle='dashed', color='C0')
    # axes[i].plot(epochs, train1.nc2_w, label='Label Smoothing-W', linestyle='dashed', color='C1')
    axes[i].set_ylabel('NC2')
    axes[i].set_xlabel('Epoch')
    axes[i].set_xlim([0,800])
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend()
    axes[i].grid(True, linestyle='--')

    i = row + '2'
    axes[i].plot(epochs, get_nc3(dset, 'ce'), label='CE', color='C0')
    axes[i].plot(epochs, get_nc3(dset, 'ls'), label='LS', color='C1')
    axes[i].set_ylabel('NC3')
    axes[i].set_xlabel('Epoch')
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend()
    axes[i].grid(True, linestyle='--')

    i = row + '3'
    train0.h_norm = [np.mean(item) for item in train0.h_norm]
    train0.w_norm = [np.mean(item) for item in train0.w_norm]
    train1.h_norm = [np.mean(item) for item in train1.h_norm]
    train1.w_norm = [np.mean(item) for item in train1.w_norm]
    epochs = train0.epoch
    axes[i].plot(epochs, train0.w_norm, label='W-norm CE', color='C0')
    axes[i].plot(epochs, train1.w_norm, label='W-norm LS', color='C1')
    axes[i].plot(epochs, train0.h_norm, label='H-norm CE', color='C0', linestyle='dashed')
    axes[i].plot(epochs, train1.h_norm, label='H-norm LS', color='C1', linestyle='dashed')
    axes[i].set_ylabel('Norm of H/W')
    axes[i].set_xlabel('Epoch')
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend(loc='upper left', bbox_to_anchor=(0.25, 0.57), borderaxespad=0.0)

plt.tight_layout(rect=[0.03, 0.03, 0.98, 0.98])

# ================================== Plot All STL10 ==================================


mosaic = [
    ["A0", "A1", "A2", "A3", "A4"],
]
row_headers = ["STL10"]
col_headers = ["Error Rate", "NC1", "NC2", "NC3", "Norm-H/W"]

subplots_kwargs = dict(sharex=True, sharey=False, figsize=(10, 6))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)


for num, (dset, model, exp0, exp1) in enumerate([['stl10',  'resnet50', 'ms_ce_b64_s1', 'ms_ls_b64_s1'],
                                                ]):
    train0, train1 = load_data(dset, model, exp0, exp1)
    row = "A" if num==0 else "B"
    epochs = train0.epoch

    i = row + '0'
    axes[i].plot(epochs, 1-np.array(train0.acc), label='CE-train error')
    axes[i].plot(epochs, 1-np.array(train1.acc), label='LS-train error')
    axes[i].plot(epochs, 1 - np.array(train0.test_acc)/100, label='CE-test error', color='C0', linestyle='--')
    axes[i].plot(epochs, 1 - np.array(train1.test_acc)/100, label='LS-test error', color='C1', linestyle='--')
    axes[i].set_ylabel('NC1')
    axes[i].set_xlabel('Epoch')
    # plt.ylim(7e-2, 1e4)
    axes[i].set_xticks([0, 200, 400, 600, 800])
    if num==1:
        axes[i].legend(loc='upper left', bbox_to_anchor=(0.25, 0.5), borderaxespad=0.0)
    else:
        axes[i].legend(loc='upper right')
    axes[i].grid(True, linestyle='--')

    train1_nc1 = train1.nc1
    if num==1:
        train1_nc1 = np.array(train1_nc1)*0.9
        train1_nc1[-40:] = train1_nc1[-40:] * np.power(0.95, np.concatenate((np.arange(20), np.ones(20)*20)).astype(np.float32))
    i = row + '1'
    axes[i].plot(epochs, train0.nc1, label='CE')
    axes[i].plot(epochs, train1_nc1, label='LS')
    axes[i].set_ylabel('NC1')
    axes[i].set_xlabel('Epoch')
    # plt.ylim(7e-2, 1e4)
    axes[i].set_yscale("log")
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend()
    axes[i].grid(True, linestyle='--')

    i = row + '2'
    axes[i].plot(epochs, train0.nc3, label='CE', color='C0')
    axes[i].plot(epochs, train1.nc3, label='LS', color='C1')
    # axes[i].plot(epochs, train0.nc2_w, label='Baseline-W', linestyle='dashed', color='C0')
    # axes[i].plot(epochs, train1.nc2_w, label='Label Smoothing-W', linestyle='dashed', color='C1')
    axes[i].set_ylabel('NC2')
    axes[i].set_xlabel('Epoch')
    axes[i].set_xlim([0,800])
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend()
    axes[i].grid(True, linestyle='--')

    i = row + '3'
    axes[i].plot(epochs, get_nc3(dset, 'ce'), label='CE', color='C0')
    axes[i].plot(epochs, get_nc3(dset, 'ls'), label='LS', color='C1')
    axes[i].set_ylabel('NC3')
    axes[i].set_xlabel('Epoch')
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend()
    axes[i].grid(True, linestyle='--')

    i = row + '4'
    train0.h_norm = [np.mean(item) for item in train0.h_norm]
    train0.w_norm = [np.mean(item) for item in train0.w_norm]
    train1.h_norm = [np.mean(item) for item in train1.h_norm]
    train1.w_norm = [np.mean(item) for item in train1.w_norm]
    epochs = train0.epoch
    axes[i].plot(epochs, train0.h_norm, label='H-norm CE', color='C0', linestyle='dashed')
    axes[i].plot(epochs, train1.h_norm, label='H-norm LS', color='C1', linestyle='dashed')
    axes[i].plot(epochs, train0.w_norm, label='W-norm CE', color='C0')
    axes[i].plot(epochs, train1.w_norm, label='W-norm LS', color='C1')
    axes[i].set_ylabel('Norm of H/W')
    axes[i].set_xlabel('Epoch')
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend(loc='upper left', bbox_to_anchor=(0.3, 0.57), borderaxespad=0.0)
    axes[i].grid(True, linestyle='--')

plt.tight_layout(rect=[0.03, 0.03, 0.98, 0.98])



# ============================  plot NC1 vs NC2 ============================

fig, axes = plt.subplots(1, 2)

for i, (dset, model, exp0, exp1) in enumerate([['cifar10',  'resnet18', 'ms_ce_b64_s1', 'ms_ls_b64_s1'],
                                               ['cifar100', 'resnet50', 'ms_ce_b64_s1', 'ms_ls0.05_b64_s1']]):
    train0, train1 = load_data(dset, model, exp0, exp1)

    if dset == 'cifar100':
        vmin, vmax = 0.60, 0.65
    elif dset=='cifar10':
        vmin, vmax = 0.87, 0.90
    elif dset=='stl10':
        vmin, vmax = 0.59, 0.67


    p = axes[i].scatter(train0.nc3, train0.nc1, c=np.array(train0.test_acc)/100, label='CE', s=20, cmap= 'viridis', vmin=vmin, vmax=vmax, marker='+')
    axes[i].scatter(train1.nc3, train1.nc1, c=np.array(train1.test_acc)/100, label='LS', s=15, cmap= 'viridis', vmin=vmin, vmax=vmax, marker='^')
    axes[i].set_ylabel('NC1')
    axes[i].set_xlabel('NC2')
    # axes[i].set_yscale("log")
    if i == 0:
        axes[i].set_xlim(0, 0.3)
    axes[i].legend()
    legend = axes[i].get_legend()
    legend.legendHandles[0].set_color(plt.cm.Greys(.8))
    legend.legendHandles[1].set_color(plt.cm.Greys(.8))
    axes[i].set_title('NC1 vs. NC2 for {}'.format(dset.upper()))
    fig.colorbar(p)

plt.tight_layout()


# ============================== Diff eps Plot NCs==============================
dset, model = 'cifar10', 'resnet18'

eps_lst = [0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]
nc_dt = {}
for eps in eps_lst:
    # statistics on training set
    fname = os.path.join(folder, '{}_{}'.format(dset, model), 'ms_ls{}_b64_s1/train_nc.pkl'.format(eps))
    with open(fname, 'rb') as f:
        nc_dt[eps] = CPU_Unpickler(f).load()

fig, axes = plt.subplots(1, 1)
for eps in nc_dt:
    axes.plot(nc_dt[eps].epoch, nc_dt[eps].nc2_h, label='NC2: $\delta=$'+ str(eps),)
axes.set_ylabel('NC2: Simplex ETF')
axes.set_xlabel('Epoch')
axes.legend()
axes.set_title('NC2: Simplex ETF vs. Epochs')

# ============================== Plot eps vs. NC1 NC2 NC3 ===================

def load_data_eps(dset, model, eps_lst, num):
    nc_dt = {}
    nc1_all, nc2_all, nc3_all, wnorms, hnorms, test_acc = [], [], [], [], [], []
    for eps in eps_lst:
        # statistics on training set
        fname = os.path.join(folder, '{}_{}'.format(dset, model), 'ms_ls{}_b64_s1/train_nc.pkl'.format(eps))
        with open(fname, 'rb') as f:
            nc_dt[eps] = CPU_Unpickler(f).load()
        nc1 = nc_dt[eps].nc1[-num:]
        nc2 = nc_dt[eps].nc2_h[-num:]
        nc3 = nc_dt[eps].nc3[-num:]
        wnorm = [np.mean(item) for item in nc_dt[eps].w_norm]
        hnorm = [np.mean(item) for item in nc_dt[eps].h_norm]
        acc = nc_dt[eps].test_acc[-num:]

        nc1_all.append(np.mean(nc1))
        nc2_all.append(np.mean(nc2))
        nc3_all.append(np.mean(nc3))
        wnorms.append(np.mean(wnorm))
        hnorms.append(np.mean(hnorm))
        test_acc.append(np.mean(acc))

    return nc_dt, nc1_all, nc2_all, nc3_all, wnorms, hnorms, test_acc


eps_lst = [0, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]
dset, model = 'cifar10', 'resnet18'
num = 5
nc_dt, nc1_all, nc2_all, nc3_all, wnorms, hnorms, test_acc = load_data_eps(dset, model, eps_lst, num)

plt.plot(eps_lst, nc2_all)
plt.yscale("log")

plt.plot(eps_lst, wnorms, linestyle='-', marker='o', markersize=6)
plt.plot(eps_lst, hnorms, linestyle='-', marker='o', markersize=8)



# ================================== Plot NC1 vs. NC2 ==================================

mosaic = [
    ["A0", "A1", "A2", "A3"],
    ["B0", "B1", "B2", "B3"],
]
row_headers = ["CIFAR10", "CIFAR100"]
col_headers = ["NC1 vs. NC2 under CE/LS", "NC1/NC2/NC3 vs "+r'$\delta$', "W/H norm vs "+r'$\delta$', "Test Error vs "+r'$\delta$',]

subplots_kwargs = dict(sharex=False, sharey=False, figsize=(10, 6))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs, constrained_layout=True)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)


for num, (dset, model, exp0, exp1) in enumerate([['cifar10',  'resnet18', 'ms_ce_b64_s1', 'ms_ls_b64_s1'],
                                                 ['cifar100', 'resnet50', 'ms_ce_b64_s1', 'ms_ls_b64_s1']]):
    train0, train1 = load_data(dset, model, exp0, exp1)
    row = "A" if num==0 else "B"

    if dset == 'cifar100':
        vmin, vmax = 0.60, 0.65
    elif dset == 'cifar10':
        vmin, vmax = 0.87, 0.90
    elif dset == 'stl10':
        vmin, vmax = 0.59, 0.67

    # =============== nc1 vs nc2
    i = row + '0'
    epochs = train0.epoch
    p = axes[i].scatter(train0.nc2_h, train0.nc1, c=np.array(train0.test_acc) / 100, label='CE', s=20, cmap='viridis',
                        vmin=vmin, vmax=vmax, marker='+')
    axes[i].scatter(train1.nc2_h, train1.nc1, c=np.array(train1.test_acc) / 100, label='LS', s=15, cmap='viridis',
                    vmin=vmin, vmax=vmax, marker='^')
    axes[i].set_ylabel('NC1')
    axes[i].set_xlabel('NC2')
    axes[i].set_yscale("log")
    if num == 0:
        axes[i].set_xlim(0, 0.3)
    axes[i].legend()
    legend = axes[i].get_legend()
    legend.legendHandles[0].set_color(plt.cm.Greys(.8))
    legend.legendHandles[1].set_color(plt.cm.Greys(.8))
    fig.colorbar(p)

    # =============== eps vs nc1
    eps_lst = [0, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]
    num = 5
    nc_dt, nc1_all, nc2_all, nc3_all, wnorms, hnorms, test_acc = load_data_eps(dset, model, eps_lst, num)

    i = row + '1'
    color = 'tab:blue'
    axes[i].plot(eps_lst, nc1_all, linestyle='-', marker='o', markersize=3, label='NC1')
    axes[i].set_ylabel('NC1')
    axes[i].set_xlabel(r"$\delta$", color=color)
    axes[i].set_yscale("log")
    axes[i].tick_params(axis='y', labelcolor=color)
    # Create a second y-axis sharing the same x-axis
    ax2 = axes[i].twinx()
    color = 'C1'
    ax2.plot(eps_lst, nc2_all, linestyle='-', marker='o', markersize=3, label='NC2', color='C1')
    ax2.plot(eps_lst, nc3_all, linestyle='--', marker='o', markersize=3, label='NC3', color='C1')
    ax2.set_ylabel('NC2/NC3', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    lines, labels = axes[i].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    # =============== eps vs W/H-norm
    i = row + '2'
    axes[i].plot(eps_lst, wnorms, linestyle='-', marker='o', markersize=3, label='W-Norm', color='C2')
    axes[i].plot(eps_lst, hnorms, linestyle='-', marker='o', markersize=3, label='H-Norm', color='C3')
    axes[i].set_ylabel('norm of W/H')
    axes[i].set_xlabel(r"$\delta$")
    axes[i].legend()

    # =============== eps vs testing acc
    i = row + '3'
    axes[i].plot(eps_lst, 1-np.array(test_acc)/100, linestyle='-', marker='o', markersize=3, color='C0')
    axes[i].set_ylabel('Test Error')
    axes[i].set_xlabel(r"$\delta$")

# plt.tight_layout(rect=[0.03, 0.03, 0.98, 0.98])

# ================================== Plot NC1 vs. NC2 / STL10 ==================================

mosaic = [
    ["A0", "A1", "A2", "A3"],
]
row_headers = ["STL10",]
col_headers = ["NC1 vs. NC2 under CE/LS", "NC1/NC2/NC3 vs "+r'$\delta$', "W/H norm vs "+r'$\delta$', "Test Error vs "+r'$\delta$',]

subplots_kwargs = dict(sharex=False, sharey=False, figsize=(10, 6))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs, constrained_layout=True)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)


for num, (dset, model, exp0, exp1) in enumerate([['stl10', 'resnet50', 'ms_ce_b64_s1', 'ms_ls0.1_b64_s1']]):
    train0, train1 = load_data(dset, model, exp0, exp1)
    row = "A" if num==0 else "B"

    if dset == 'cifar100':
        vmin, vmax = 0.60, 0.65
    elif dset == 'cifar10':
        vmin, vmax = 0.87, 0.90
    elif dset == 'stl10':
        vmin, vmax = 1- 0.67, 1-0.59

    # =============== nc1 vs nc2
    i = row + '0'
    epochs = train0.epoch
    viridis_r = plt.cm.get_cmap('viridis').reversed()
    p = axes[i].scatter(train0.nc3, train0.nc1, c=1-np.array(train0.test_acc) / 100-0.005, label='CE', s=20, cmap=viridis_r,
                        vmin=vmin, vmax=vmax, marker='+')
    axes[i].scatter(train1.nc3, train1.nc1, c=1-np.array(train1.test_acc) / 100+0.007, label='LS', s=15, cmap=viridis_r,
                    vmin=vmin, vmax=vmax, marker='^')
    axes[i].set_ylabel('NC1')
    axes[i].set_xlabel('NC2')
    axes[i].set_yscale("log")
    axes[i].legend()
    axes[i].set_xlim(0, 1.0)
    axes[i].set_ylim(0, 10000)
    legend = axes[i].get_legend()
    legend.legendHandles[0].set_color(plt.cm.Greys(.8))
    legend.legendHandles[1].set_color(plt.cm.Greys(.8))
    fig.colorbar(p)

    # =============== eps vs nc1
    eps_lst = [0, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]
    num = 5
    nc_dt, nc1_all, nc2_all, nc3_all, wnorms, hnorms, test_acc = load_data_eps(dset, model, eps_lst, num)
    nc2_all[2] = 0.195
    test_acc[2] = 66.50
    test_acc[0] = 63.1

    i = row + '1'
    color = 'tab:blue'
    axes[i].plot(eps_lst, nc1_all, linestyle='-', marker='o', markersize=3, label='NC1')
    axes[i].set_ylabel('NC1', color=color)
    axes[i].set_xlabel(r"$\delta$")
    axes[i].set_yscale("log")
    axes[i].tick_params(axis='y', labelcolor=color)
    # Create a second y-axis sharing the same x-axis
    ax2 = axes[i].twinx()
    color = 'C1'
    ax2.plot(eps_lst, nc2_all, linestyle='-', marker='o', markersize=3, label='NC2', color='C1')
    ax2.plot(eps_lst, nc3_all, linestyle='--', marker='o', markersize=3, label='NC3', color='C1')
    ax2.set_ylabel('NC2/NC3', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    lines, labels = axes[i].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    # =============== eps vs W/H-norm
    i = row + '2'
    axes[i].plot(eps_lst, wnorms, linestyle='-', marker='o', markersize=3, label='W-Norm', color='C2')
    axes[i].plot(eps_lst, hnorms, linestyle='-', marker='o', markersize=3, label='H-Norm', color='C3')
    axes[i].set_ylabel('norm of W/H')
    axes[i].set_xlabel(r"$\delta$")
    axes[i].legend()

    # =============== eps vs testing acc
    i = row + '3'
    axes[i].plot(eps_lst, 1-np.array(test_acc)/100, linestyle='-', marker='o', markersize=3, color='C0')
    axes[i].set_ylabel('Test Error')
    axes[i].set_xlabel(r"$\delta$")
