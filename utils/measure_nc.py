# -*- coding: utf-8 -*-
import os
import torch
import random
import numpy as np
from scipy.sparse.linalg import svds


def analysis(model, loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N      = [0 for _ in range(args.num_classes)]   # within class sample size
    mean   = [0 for _ in range(args.num_classes)]
    Sw_cls = [0 for _ in range(args.num_classes)]
    loss = 0
    n_correct = 0

    model.eval()
    criterion_summed = torch.nn.CrossEntropyLoss(reduction='sum')
    for computation in ['Mean', 'Cov']:
        for batch_idx, (data, target) in enumerate(loader, start=1):
            if isinstance(data, list):
                data = data[0]
            data, target = data.to(device), target.to(device)

            with torch.no_grad():
                output, h = model(data, ret='of')  # [B, C], [B, 512]

            for c in range(args.num_classes):
                idxs = (target == c).nonzero(as_tuple=True)[0]
                if len(idxs) == 0:  # If no class-c in this batch
                    continue
                h_c = h[idxs, :]    # [B, 512]

                # update class means
                if computation == 'Mean':
                    N[c] += h_c.shape[0]
                    mean[c] += torch.sum(h_c, dim=0)
                    loss += criterion_summed(output, target).item()  # during calculation of class means, calculate loss

                # update within-class cov
                elif computation == 'Cov':
                    z = h_c - mean[c].unsqueeze(0)  # [B, 512]
                    cov = torch.matmul(z.unsqueeze(-1), z.unsqueeze(1))   # [B 512 1] [B 1 512] -> [B, 512, 512]
                    Sw_cls[c] += torch.sum(cov, dim=0)     # [512, 512]

                    # during calculation of within-class covariance, calculate network's accuracy
                    net_pred = torch.argmax(output[idxs, :], dim=1)
                    n_correct += sum(net_pred == target[idxs]).item()

        if computation == 'Mean':
            for c in range(args.num_classes):
                mean[c] /= N[c]
            M = torch.stack(mean).T
        elif computation == 'Cov':
            Sw_all = sum(Sw_cls)           # [512, 512]
            for c in range(args.num_classes):
                Sw_cls[c] = Sw_cls/N[c]

    loss /= sum(N)
    acc = n_correct / sum(N)

    # between-class covariance
    muG = torch.mean(M, dim=1, keepdim=True)  # global mean: [512, C]
    M_ = M - muG  # [512, C]
    Sb = torch.matmul(M_, M_.T) / args.num_classes

    # nc1 = tr{Sw Sb^-1}
    Sw = Sw_all.cpu().numpy()
    Sb = Sb.cpu().numpy()
    eigvec, eigval, _ = svds(Sb, k=args.num_classes - 1)
    inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T
    nc1 = np.trace(Sw @ inv_Sb)
    nc1_cls = [np.trace(Sw_cls1.cpu().numpy() @ inv_Sb) for Sw_cls1 in Sw_cls]
    nc1_cls = np.array(nc1_cls)

    # avg norm
    W = model.fc_cb.weight.detach()       # [C, 512]
    M_norms = torch.norm(M_, dim=0)   # [C]
    W_norms = torch.norm(W.T, dim=0)  # [C]
    #h_norm_cov = (torch.std(M_norms) / torch.mean(M_norms)).item()
    #w_norm_cov = (torch.std(W_norms) / torch.mean(W_norms)).item()

    # mutual coherence
    W_nomarlized = W.T / W_norms   # [512, C]
    cos = ( W_nomarlized.T @ W_nomarlized ).cpu().numpy()  # [C, D] [D, C] -> [C, C]
    cos_avg = (cos.sum(1) - np.diag(cos)) / (cos.shape[1] - 1)

    # angle between H
    M_normalized = M_ / M_norms  # [512, C]
    h_cos = (M_normalized.T @ M_normalized).cpu().numpy()
    h_cos_avg = (h_cos.sum(1)-np.diag(h_cos)) / (h_cos.shape[1]-1)

    # angle between W and H
    cos_wh = torch.sum(W_nomarlized*M_normalized, dim=0).cpu().numpy()  # [C]

    return {
        "loss": loss,
        "acc": acc,
        "nc1": nc1,
        "nc1_cls": nc1_cls,
        "w_norm": W_norms.cpu().numpy(),
        "h_norm": M_norms.cpu().numpy(),
        "w_cos": cos,
        "w_cos_avg": cos_avg,
        "h_cos":h_cos,
        "h_cos_avg": h_cos_avg,
        "wh_cos": cos_wh
    }

