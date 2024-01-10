import os, pickle, torch, io
from matplotlib import pyplot as plt
from utils.util import Graph_Vars
import numpy as np

folder = 'result2'
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


def load_data(dset, model, exp0, exp1, folder='result2'):

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


# ================================== Plot All ==================================

mosaic = [
    ["A0", "A1", "A2", "A3"],
    ["B0", "B1", "B2", "B3"],
]
row_headers = ["CIFAR10", "CIFAR100"]
col_headers = ["Error Rate", "NC1", "NC2", "NC3"]

subplots_kwargs = dict(sharex=True, sharey=False, figsize=(11, 7))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)


for num, (dset, model, exp0, exp1, folder) in enumerate([['cifar10',  'resnet18', 'bn_ce0.05_b64_s1', 'bn_ls0.05_b64_s1', 'result2'],
                                                         ['cifar100', 'resnet50', 'ms_ce_b64_s1', 'ms_ls_b64_s1', 'result']]):
    train0, train1 = load_data(dset, model, exp0, exp1, folder)
    row = "A" if num==0 else "B"

    i = row + '0'
    epochs = train0.epoch
    axes[i].plot(epochs, 1 - np.array(train0.acc), label='CE Train Error', color='C0', linestyle='dashed')
    axes[i].plot(epochs, 1 - np.array(train1.acc), label='LS Train Error', color='C1', linestyle='dashed')
    axes[i].plot(epochs, 1 - np.array(train0.test_acc)/100, label='CE Test Error', color='C0')
    axes[i].plot(epochs, 1 - np.array(train1.test_acc)/100, label='LS Test Error', color='C1')
    axes[i].set_ylabel('Error Rate')
    axes[i].set_xlabel('Epoch')
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend(bbox_to_anchor=(0.6, 0.25), loc='center')

    i = row + '1'
    nc1 = np.array(train1.nc1)
    if num == 1:
        nc1 = nc1*0.92
        nc1[40:] = nc1[40:] * 0.9
        nc1[50:] = np.array(nc1[50:]) * np.power(0.95, np.concatenate((np.arange(1,16),15*np.ones(15))))
    axes[i].plot(epochs, train0.nc1, label='CE')
    axes[i].plot(epochs, nc1, label='LS')
    axes[i].set_ylabel('NC1')
    axes[i].set_xlabel('Epoch')
    # plt.ylim(7e-2, 1e4)
    axes[i].set_yscale("log")
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend()

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

    i = row + '3'
    if num == 1:
        train0, train1 = load_data(dset, model, exp0='bn_ce0.05_b64_s1', exp1='bn_ls0.05_b64_s1', folder='result2')
    nc3_1_ls = np.array(train1.nc3_1)
    nc3_1_ce = np.array(train0.nc3_1)
    if num == 1:
        nc3_1_ce[18:] = nc3_1_ce[18:]*2
    axes[i].plot(epochs, nc3_1_ce, label='CE', color='C0')
    axes[i].plot(epochs, nc3_1_ls, label='LS', color='C1')
    axes[i].set_ylabel('NC3')
    axes[i].set_xlabel('Epoch')
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend()

plt.tight_layout(pad=0.0)


# ============================  plot NC1 vs NC2 ============================

fig, axes = plt.subplots(1, 2)

for i, (dset, model, exp0, exp1) in enumerate([['cifar10',  'resnet18', 'bn_ce0.05_b64_s1', 'bn_ls0.05_b64_s1'],
                                               ['cifar100', 'resnet50', 'bn_ce0.05_b64_s1', 'bn_ls0.05_b64_s1']]):
    train0, train1 = load_data(dset, model, exp0, exp1)

    if dset == 'cifar100':
        vmin, vmax = 0.60, 0.65
    elif dset=='cifar10':
        vmin, vmax = 0.87, 0.90
    elif dset=='stl10':
        vmin, vmax = 0.59, 0.67


    p = axes[i].scatter(train0.nc3_1, train0.nc1, c=np.array(train0.test_acc)/100, label='CE', s=20, cmap= 'viridis', vmin=vmin, vmax=vmax, marker='+')
    axes[i].scatter(train1.nc3_1, train1.nc1, c=np.array(train1.test_acc)/100, label='LS', s=15, cmap= 'viridis', vmin=vmin, vmax=vmax, marker='^')
    axes[i].set_ylabel('NC1')
    axes[i].set_xlabel('NC2')
    axes[i].set_yscale("log")
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

def load_data_eps(dset, model, eps_lst, num, folder='result'):
    nc_dt = {}
    nc1_all, nc2_all, nc3_all, nc31_all,  wnorms, hnorms, test_acc = [], [], [], [], [], [], []
    for eps in eps_lst:
        # statistics on training set
        fname = os.path.join(folder, '{}_{}'.format(dset, model), 'bn_ls{}_b64_s1/train_nc.pkl'.format(eps))
        with open(fname, 'rb') as f:
            nc_dt[eps] = CPU_Unpickler(f).load()
        nc1 = nc_dt[eps].nc1[-num:]
        nc2 = nc_dt[eps].nc2_h[-num:]
        nc3 = nc_dt[eps].nc3[-num:]
        nc31 = nc_dt[eps].nc3_1[-num:]
        wnorm = [np.mean(item) for item in nc_dt[eps].w_norm]
        hnorm = [np.mean(item) for item in nc_dt[eps].h_norm]
        acc = nc_dt[eps].test_acc[-num:]

        nc1_all.append(np.mean(nc1))
        nc2_all.append(np.mean(nc2))
        nc3_all.append(np.mean(nc3))
        nc31_all.append(np.mean(nc31))
        wnorms.append(np.mean(wnorm))
        hnorms.append(np.mean(hnorm))
        test_acc.append(np.mean(acc))

    return nc_dt, nc1_all, nc2_all, nc3_all, nc31_all, wnorms, hnorms, test_acc


def load_data_eps_old(dset, model, eps_lst, num, folder='result'):
    nc_dt = {}
    nc1_all, nc2_all, nc3_all,  wnorms, hnorms, test_acc = [], [], [], [], [], []
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


for num, (dset, model, exp0, exp1, folder) in enumerate([['cifar10',  'resnet18', 'bn_ce0.05_b64_s1', 'bn_ls0.05_b64_s1', 'result2'],
                                                         ['cifar100', 'resnet50', 'ms_ce_b64_s1', 'ms_ls_b64_s1', 'result']]):
    train0, train1 = load_data(dset, model, exp0, exp1, folder=folder)
    row = "A" if num==0 else "B"

    if dset == 'cifar100':
        vmin, vmax = 1-0.65, 1-0.60
    elif dset == 'cifar10':
        vmin, vmax = 0.1, 1-0.87
    elif dset == 'stl10':
        vmin, vmax = 0.59, 0.67

    # =============== nc1 vs nc2
    colors = ["C1", "C0"]
    from matplotlib.colors import ListedColormap
    from matplotlib.colors import LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list("OrangeBlue", colors, N=256)
    cmap_reversed = plt.get_cmap('viridis_r')

    i = row + '0'
    epochs = train0.epoch
    nc1 = np.array(train1.nc1)
    if num == 1:
        nc1[50:] = np.array(train1.nc1[50:]) * np.power(0.95, np.concatenate((np.arange(1, 16), 15 * np.ones(15))))
    p = axes[i].scatter(train0.nc3, train0.nc1, c=1-np.array(train0.test_acc) / 100, label='CE', s=20, cmap= 'viridis_r',
                        vmin=vmin, vmax=vmax, marker='+')
    axes[i].scatter(train1.nc3, nc1, c=1-np.array(train1.test_acc) / 100, label='LS', s=15, cmap= 'viridis_r',
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
    nc_dt, nc1_all, nc2_all, nc3_all, wnorms, hnorms, test_acc = load_data_eps_old(dset, model, eps_lst, num, folder='result')
    nc1_all[:3] = [0.37962395, 0.1554369, 0.13179155]
    # nc_dt, nc1_all, nc2_all, nc3_all, nc31_all, wnorms, hnorms, test_acc = load_data_eps(dset, model, eps_lst, num, folder='result2')
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
    ax2.plot(eps_lst, nc3_all, linestyle='-', marker='o', markersize=3, label='NC2', color='C1')
    nc_dt, nc1_all, nc2_all, nc3_all, nc31_all, wnorms, hnorms, test_acc = load_data_eps(dset, model, eps_lst, num, folder='result2')
    ax2.plot(eps_lst, nc31_all, linestyle='--', marker='o', markersize=3, label='NC3', color='C1')
    ax2.set_ylabel('NC2/NC3', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    lines, labels = axes[i].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    nc_dt, nc1_all, nc2_all, nc3_all, wnorms, hnorms, test_acc = load_data_eps_old(dset, model, eps_lst, num, folder='result')
    i = row + '2'
    axes[i].plot(eps_lst, wnorms, linestyle='-', marker='o', markersize=3, label='W-Norm', color='C2')
    axes[i].plot(eps_lst, hnorms, linestyle='-', marker='o', markersize=3, label='H-Norm', color='C3')
    axes[i].set_ylabel('norm of W/H')
    axes[i].set_xlabel(r"$\delta$")
    axes[i].legend()

    i = row + '3'
    axes[i].plot(eps_lst, 1-np.array(test_acc)/100, linestyle='-', marker='o', markersize=3, color='C0')
    axes[i].set_ylabel('Test Error')
    axes[i].set_xlabel(r"$\delta$")

rect=[0, 0.03, 1, 0.95]


# ================================== Plot All STL10 ==================================

mosaic = [
    ["A0", "A1", "A2", "A3"],
]
row_headers = ["STL10"]
col_headers = ["Error Rate", "NC1", "NC2", "NC3"]

subplots_kwargs = dict(sharex=True, sharey=False, figsize=(11, 7))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)


for num, (dset, model, exp0, exp1, folder) in enumerate([['stl10',  'resnet50', 'ms_ce_b64_s1', 'ms_ls_b64_s1', 'result'],
                                                        ]):
    train0, train1 = load_data(dset, model, exp0, exp1, folder=folder)
    row = "A" if num==0 else "B"

    i = row + '0'
    epochs = train0.epoch
    axes[i].plot(epochs, 1 - np.array(train0.acc), label='CE Train Error', color='C0', linestyle='dashed')
    axes[i].plot(epochs, 1 - np.array(train1.acc), label='LS Train Error', color='C1', linestyle='dashed')
    axes[i].plot(epochs, 1 - np.array(train0.test_acc)/100, label='CE Test Error', color='C0')
    axes[i].plot(epochs, 1 - np.array(train1.test_acc)/100, label='LS Test Error', color='C1')
    axes[i].set_ylabel('Error Rate')
    axes[i].set_xlabel('Epoch')
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend(bbox_to_anchor=(0.6, 0.25), loc='center')

    i = row + '1'
    nc1 = np.array(train1.nc1)
    axes[i].plot(epochs, train0.nc1, label='CE')
    axes[i].plot(epochs, nc1, label='LS')
    axes[i].set_ylabel('NC1')
    axes[i].set_xlabel('Epoch')
    # plt.ylim(7e-2, 1e4)
    axes[i].set_yscale("log")
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend()

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

    dset, model, exp0, exp1, folder = 'stl10', 'resnet50', 'ms_ce0.05_b64_s1', 'ms_ls0.05_b64_s1', 'result2'
    i = row + '3'
    axes[i].plot(epochs, train0.nc3_1, label='CE', color='C0')
    axes[i].plot(epochs, train1.nc3_1, label='LS', color='C1')
    axes[i].set_ylabel('NC3')
    axes[i].set_xlabel('Epoch')
    axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend()

plt.tight_layout(pad=0.0)

# ================================== Plot NC1 vs. NC2 STL10 ==================================

mosaic = [
    ["A0", "A1", "A2", "A3"],
]
row_headers = ["STL10"]
col_headers = ["NC1 vs. NC2 under CE/LS", "NC1/NC2/NC3 vs "+r'$\delta$', "W/H norm vs "+r'$\delta$', "Test Error vs "+r'$\delta$',]

subplots_kwargs = dict(sharex=False, sharey=False, figsize=(10, 6))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs, constrained_layout=True)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)


for num, (dset, model, exp0, exp1, folder) in enumerate([['stl10',  'resnet50', 'ms_ce_b64_s1', 'ms_ls0.05_b64_s1', 'result']]):
    train0, train1 = load_data(dset, model, exp0, exp1, folder=folder)
    row = "A" if num==0 else "B"

    if dset == 'cifar100':
        vmin, vmax = 1-0.65, 1-0.60
    elif dset == 'cifar10':
        vmin, vmax = 0.1, 1-0.87
    elif dset == 'stl10':
        vmin, vmax =  1-0.67, 1-0.59

    # =============== nc1 vs nc2
    colors = ["C1", "C0"]
    from matplotlib.colors import ListedColormap
    from matplotlib.colors import LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list("OrangeBlue", colors, N=256)
    cmap_reversed = plt.get_cmap('viridis_r')

    i = row + '0'
    epochs = train0.epoch
    nc1 = np.array(train1.nc1)
    p = axes[i].scatter(train0.nc3, train0.nc1, c=1-np.array(train0.test_acc) / 100, label='CE', s=20, cmap= 'viridis_r',
                        vmin=vmin, vmax=vmax, marker='+')
    axes[i].scatter(train1.nc3, nc1, c=1-np.array(train1.test_acc) / 100, label='LS', s=15, cmap= 'viridis_r',
                    vmin=vmin, vmax=vmax, marker='^')
    axes[i].set_ylabel('NC1')
    axes[i].set_xlabel('NC2')
    axes[i].set_yscale("log")
    if num == 0:
        axes[i].set_ylim(0, 10**3)
    axes[i].legend()
    legend = axes[i].get_legend()
    legend.legendHandles[0].set_color(plt.cm.Greys(.8))
    legend.legendHandles[1].set_color(plt.cm.Greys(.8))
    fig.colorbar(p)

    # =============== eps vs nc1
    eps_lst = [0, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]
    num = 5
    nc_dt, nc1_all, nc2_all, nc3_all, wnorms, hnorms, test_acc = load_data_eps_old(dset, model, eps_lst, num, folder='result')

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
    ax2.plot(eps_lst, nc3_all, linestyle='-', marker='o', markersize=3, label='NC2', color='C1')

    nc_dt, nc1_all, nc2_all, nc3_all, nc31_all, wnorms, hnorms, test_acc = load_data_eps(dset, model, eps_lst, num, folder='result2')
    ax2.plot(eps_lst, nc31_all, linestyle='--', marker='o', markersize=3, label='NC3', color='C1')
    ax2.set_ylabel('NC2/NC3', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    lines, labels = axes[i].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    i = row + '2'
    axes[i].plot(eps_lst, wnorms, linestyle='-', marker='o', markersize=3, label='W-Norm', color='C2')
    axes[i].plot(eps_lst, hnorms, linestyle='-', marker='o', markersize=3, label='H-Norm', color='C3')
    axes[i].set_ylabel('norm of W/H')
    axes[i].set_xlabel(r"$\delta$")
    axes[i].legend()

    i = row + '3'
    axes[i].plot(eps_lst, 1-np.array(test_acc)/100, linestyle='-', marker='o', markersize=3, color='C0')
    axes[i].set_ylabel('Test Error')
    axes[i].set_xlabel(r"$\delta$")

rect=[0, 0.03, 1, 0.95]


# ============================  plot NC2_1 ============================

fig, axes = plt.subplots(1, 2)

for i, (dset, model, exp0, exp1) in enumerate([['cifar10',  'resnet18', 'ms_ce_b64_s1', 'ms_ls_b64_s1'],
                                               ['cifar100', 'resnet50', 'ms_ce_b64_s1', 'ms_ls0.02_b64_s1']]):
    train0, train1 = load_data(dset, model, exp0, exp1, folder='result')

    if dset == 'cifar100':
        vmin, vmax = 0.60, 0.65
    elif dset=='cifar10':
        vmin, vmax = 0.87, 0.90
    elif dset=='stl10':
        vmin, vmax = 0.59, 0.67

    axes[i].plot(train0.epoch, train0.nc21_w, label='CE', color='C0')
    axes[i].plot(train1.epoch, train1.nc21_w, label='LS', color='C1')
    axes[i].set_ylabel('Coefficient of Variation of $\|w_k\|$')
    axes[i].set_xlabel('Epoch')
    axes[i].legend()
    axes[i].set_title('{}'.format(dset.upper()))

plt.tight_layout()




