import sys
import math
import time
import wandb
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from sklearn.metrics import confusion_matrix

from utils import util
from utils.util import *
from utils.plot import plot_nc
from utils.measure_nc import analysis, analysis_feat
from model.KNN_classifier import KNNClassifier
from model.loss import CrossEntropyLabelSmooth, CDTLoss, LDTLoss, CombinedMarginLoss


def soften_target(targets, num_classes, epsilon):
    targets = torch.zeros(targets.size(0), num_classes).scatter_(
        1,
        targets.unsqueeze(1).cpu(), 1)

    if torch.cuda.is_available(): targets = targets.cuda()
    targets = (1 - epsilon) * targets + epsilon / num_classes
    return targets


def get_samples_per_class(dataset, num_samples_per_class=10, num_classes=10):
    samples_per_class = {k: [] for k in range(num_classes)}
    for idx, (image, label) in enumerate(dataset):
        if len(samples_per_class[label]) < num_samples_per_class:
            samples_per_class[label].append(image.unsqueeze(0))
        if all(len(samples) == num_samples_per_class for samples in samples_per_class.values()):
            break
    return samples_per_class


class Trainer_bn(object):
    def __init__(self, args, model=None, train_loader=None, val_loader=None, weighted_train_loader=None,
                 per_class_num=[], log=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.print_freq = args.print_freq
        self.label_weighting = args.label_weighting
        self.num_classes = args.num_classes

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.weighted_train_loader = weighted_train_loader

        # init queue
        if args.bn_type == 'cbn0':
            samples_per_class = get_samples_per_class(train_loader.dataset,
                                                      num_samples_per_class=np.ceil(args.batch_size / args.num_classes),
                                                      num_classes=args.num_classes)
            self.queue = {k: torch.cat(samples_per_class[k], dim=0).to(self.device) for k in range(args.num_classes)}
            self.queue_ptr = {k: torch.zeros(1, dtype=torch.long) for k in range(args.num_classes)}

        self.per_cls_weights = None
        self.cls_num_list = per_class_num
        self.contrast_weight = args.contrast_weight
        self.log = log

        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.weight_decay)
        if self.args.scheduler == 'cos':
            self.train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)
        elif self.args.scheduler == 'ms':
            self.train_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[70, 140], gamma=0.1),
        self.update_weight()
        self.set_loss()

    def update_weight(self):
        per_cls_weights = 1.0 / (np.array(self.cls_num_list) ** self.label_weighting)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.cls_num_list)
        self.per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.device)

    def set_loss(self):
        if self.args.loss == 'ce':
            self.criterion = nn.CrossEntropyLoss(reduction='mean')  # train fc_bc
        elif self.args.loss == 'ls':
            self.criterion = CrossEntropyLabelSmooth(self.args.num_classes, epsilon=self.args.eps)
        elif self.args.loss == 'ldt':
            delta_list = self.cls_num_list / np.min(self.cls_num_list)
            self.criterion = LDTLoss(delta_list, gamma=0.5, device=self.device)
        elif self.args.loss == 'wce':
            self.criterion = nn.CrossEntropyLoss(reduction='mean', weight=self.per_cls_weights)
        elif self.args.loss == 'hce':
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
        elif self.args.loss == 'bce':
            self.criterion = nn.BCELoss(reduction='mean')
        elif self.args.loss == 'arcf' or self.args.loss == 'arcm':
            self.criterion = CombinedMarginLoss(64, self.args.margins[0], self.args.margins[1], self.args.margins[2])

    def train_one_epoch(self, epoch):

        # switch to train mode
        self.model.train()
        losses = AverageMeter('Loss', ':.4e')
        train_acc = AverageMeter('Train_acc', ':.4e')

        if self.args.resample_weighting > 0:
            train_loader = self.weighted_train_loader
        else:
            train_loader = self.train_loader

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            if self.args.loss == 'arcm' and self.args.bias == 'g' and epoch<=1 and i<=1 and self.args.dataset=='cifar10':
                with torch.no_grad(): 
                    _, feat = self.model(inputs, targets, ret='of')
                    self.model.fc.mu.data = torch.mean(feat, dim=0)

            if self.args.aug == 'cm' or self.args.aug == 'cutmix':  # cutmix augmentation within the mini-batch
                cutmix = v2.CutMix(num_classes=self.args.num_classes)
                inputs, reweighted_targets = cutmix(inputs, targets)  # reweighted target will be [B, K]

            # if self.args.mixup >= 0:
            #     output, reweighted_targets, h = self.model.forward_mixup(inputs, targets, mixup=self.args.mixup,
            #                                                              mixup_alpha=self.args.mixup_alpha)

            freq = torch.bincount(targets, minlength=self.args.num_classes)
            cls_idx = torch.where(freq == 0)[0]

            if len(cls_idx) > 0 and (self.args.bn_type == 'cbn' or self.args.bn_type == 'cbn0'):
                if self.args.bn_type == 'cbn0':
                    bn_inputs = torch.cat([self.queue[k] for k in cls_idx.cpu().numpy()], dim=0).to(self.device)
                    bn_targets = torch.cat([torch.tensor(k).repeat(len(self.queue[0])) for k in cls_idx.cpu().numpy()],
                                           dim=0).to(self.device)
                else:
                    try:
                        bn_inputs, bn_targets = next(weighted_train_loader)
                    except:
                        weighted_train_loader = iter(self.weighted_train_loader)
                        bn_inputs, bn_targets = next(weighted_train_loader)
                    bn_inputs, bn_targets = bn_inputs.to(self.device), bn_targets.to(self.device)

                all_inputs = torch.cat([inputs, bn_inputs])
                all_targets = torch.cat([targets, bn_targets])
            else:
                all_inputs, all_targets = inputs, targets

            output_all, h_all = self.model(all_inputs, all_targets, ret='of')
            output, h = output_all[0:len(inputs)], h_all[0:len(inputs)]

            # update the img_bank with current batch
            if self.args.bn_type == 'cbn0':
                for k in self.queue:
                    cls_idx = torch.where(targets == k)[0]
                    if len(cls_idx) == 0:
                        pass
                    else:
                        ptr = self.queue_ptr[k]
                        num_cls = min(len(self.queue[k]), len(cls_idx))

                        # replace the keys at ptr (dequeue and enqueue)
                        if ptr + num_cls <= len(self.queue[k]):
                            self.queue[k][ptr:ptr + num_cls] = inputs[cls_idx][:num_cls]
                        else:
                            self.queue[k][ptr:] = inputs[cls_idx][num_cls - (len(self.queue[k]) - ptr):num_cls]
                            self.queue[k][:num_cls - (len(self.queue[k]) - ptr)] = inputs[cls_idx][:num_cls - (
                                        len(self.queue[k]) - ptr)]
                        self.queue_ptr[k] = (ptr + num_cls) % len(self.queue[k])  # move pointer

            # ==== update loss and acc
            train_acc.update(torch.sum(output.argmax(dim=-1) == targets).item() / targets.size(0),
                             targets.size(0)
                             )
            loss = self.criterion(output,
                                  reweighted_targets if self.args.mixup >= 0 or self.args.aug == 'cm' or self.args.aug == 'cutmix' else targets)
            losses.update(loss.item(), targets.size(0))

            # ==== gradient update
            if self.args.loss != 'hce':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            elif self.args.loss == 'hce':

                # gradient of L wrt. b
                beta = self.per_cls_weights[targets]  # [B]
                P = nn.Softmax(dim=-1)(output.detach())  # [B, K]
                Y = torch.eye(self.args.num_classes, device=targets.device)[targets]  # [B, K]
                b_grad = beta.unsqueeze(1) * (P - Y)  # [B, K]
                b_grad = torch.sum(b_grad, dim=0) / len(b_grad)

                # gradient of L wrt. W
                weighted_P_Y = (P.detach() - Y) * beta.unsqueeze(1)  # [B, K]
                W_grad = torch.einsum('db, bk->dk', h.detach().T, weighted_P_Y) / len(output)  # [D, K]
                W_grad = W_grad.T  # [K, D]

                self.optimizer.zero_grad()
                loss.backward()
                self.model.fc_cb.bias.grad = b_grad
                self.model.fc_cb.weight.grad = W_grad
                self.optimizer.step()
        return losses, train_acc

    def train_base(self):
        best_acc1 = 0

        # tell wandb to watch what the model gets up to: gradients, weights, and more!
        wandb.watch(self.model, self.criterion, log=None, log_freq=10)
        train_nc = Graph_Vars()
        val_nc = Graph_Vars()

        for epoch in range(self.args.start_epoch, self.args.epochs):

            # ============ training ============
            start_time = time.time()
            losses, train_acc = self.train_one_epoch(epoch=epoch)
            epoch_time = time.time() - start_time
            self.log.info(
                '====>EPOCH{epoch}Train{iters}, Epoch Time:{epoch_time:.3f}, Loss:{loss:.4f}, Acc:{acc:.4f}'.format(
                    epoch=epoch + 1, iters=len(self.train_loader), epoch_time=epoch_time, loss=losses.avg,
                    acc=train_acc.avg
                ))
            wandb.log({'train/train_loss': losses.avg,
                       'train/train_acc': train_acc.avg,
                       'lr': self.optimizer.param_groups[0]['lr']},
                      step=epoch + 1)
            if epoch % 10 == 0 and (self.args.bias in ['t', 'true'] and self.args.loss=='ce'):
                bias_values = self.model.fc.bias.data
                self.log.info('--Epoch_{epoch}, Bias: {bias_str}'.format(
                    epoch=epoch + 1, bias_str=', '.join([f'{bias_value:.4f}' for bias_value in bias_values])))

            # ============ evaluation ============

            # === validation using Nearest centroid
            if self.args.imbalance_rate < 1.0:
                cfeats = self.get_knncentroids()
                self.knn_classifier = KNNClassifier(feat_dim=self.model.out_dim, num_classes=self.args.num_classes,
                                                    feat_type='cl2n', dist_type='l2')
                self.knn_classifier.update(cfeats)
            else:
                self.knn_classifier = None

            val_targets, val_feats, val_logits, val_ncc_logits = self.get_feat_logits(self.val_loader)

            # ==== regular validation
            acc1, acc5 = accuracy(val_logits, val_targets, topk=(1, 5))
            cls_acc, many_acc, medium_acc, few_acc = self.calculate_acc(val_targets.cpu().numpy(),
                                                                        val_logits.argmax(1).cpu().numpy())
            self.log.info('---->EPOCH {} Val: Prec@1 {:.3f} Prec@5 {:.3f}'.format(epoch, acc1, acc5))
            self.log.info("many acc {:.2f}, med acc {:.2f}, few acc {:.2f}".format(many_acc, medium_acc, few_acc))

            wandb.log({'val/val_acc1': acc1,
                       'val/val_acc5': acc5,
                       'val/val_many': many_acc,
                       'val/val_medium': medium_acc,
                       'val/val_few': few_acc},
                      step=epoch + 1)

            # === validation using Nearest centroid classifier
            if self.args.imbalance_rate < 1.0:
                acc1, acc5 = accuracy(val_ncc_logits, val_targets, topk=(1, 5))
                cls_acc, many_acc, medium_acc, few_acc = self.calculate_acc(val_targets.cpu().numpy(),
                                                                            val_ncc_logits.argmax(1).cpu().numpy())
                self.log.info('---->EPOCH {} NCC Val: Prec@1 {:.3f} Prec@5 {:.3f}'.format(epoch, acc1, acc5))
                self.log.info("many acc {:.2f}, med acc {:.2f}, few acc {:.2f}".format(many_acc, medium_acc, few_acc))

                wandb.log({'knn_val/val_acc1': acc1,
                           'knn_val/val_acc5': acc5,
                           'knn_val/val_many': many_acc,
                           'knn_val/val_medium': medium_acc,
                           'knn_val/val_few': few_acc},
                          step=epoch + 1)

            if self.args.dataset == 'ImageNet-LT' or self.args.dataset == 'iNaturelist2018':
                self.paco_adjust_learning_rate(self.optimizer, epoch, self.args)
            else:
                self.train_scheduler.step()
            self.model.train()

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            save_checkpoint(self.args,
                            {'epoch': epoch + 1, 'state_dict': self.model.state_dict(), 'best_acc1': best_acc1, },
                            is_best, epoch + 1)

            # # ============ Measure NC ============
            if self.args.debug > 0:
                if (epoch + 1) % self.args.debug == 0:
                    nc_dict = analysis(self.model, self.train_loader, self.args)
                    self.log.info('Loss:{:.3f}, Acc:{:.2f}, NC1:{:.3f}, NC2h:{:.3f}, NC2W:{:.3f}, NC3:{:.3f}'.format(
                        nc_dict['loss'], nc_dict['acc'], nc_dict['nc1'], nc_dict['nc2_h'], nc_dict['nc2_w'],
                        nc_dict['nc3'],
                    ))
                    train_nc.load_dt(nc_dict, epoch=epoch + 1, lr=self.optimizer.param_groups[0]['lr'])
                    wandb.log({'nc/loss': nc_dict['loss'],
                               'nc/acc': nc_dict['acc'],
                               'nc/nc1': nc_dict['nc1'],
                               'nc/nc2h': nc_dict['nc2_h'],
                               'nc/nc2w': nc_dict['nc2_w'],
                               'nc/nc3': nc_dict['nc3'],
                               'nc/nc3d': nc_dict['nc3_d'],
                               },
                              step=epoch + 1)
                    if self.args.imbalance_type == 'step' and self.args.imbalance_rate < 1.0:
                        wandb.log({'nc1/w_mnorm': nc_dict['w_mnorm'],
                                   'nc1/w_mnorm1': nc_dict['w_mnorm1'],
                                   'nc1/w_mnorm2': nc_dict['w_mnorm2'],
                                   'nc1/h_mnorm': nc_dict['h_mnorm'],
                                   'nc1/h_mnorm1': nc_dict['h_mnorm1'],
                                   'nc1/h_mnorm2': nc_dict['h_mnorm2'],
                                   'nc1/w_cos1': nc_dict['w_cos1'],
                                   'nc1/w_cos2': nc_dict['w_cos2'],
                                   'nc1/w_cos3': nc_dict['w_cos3'],
                                   'nc1/h_cos1': nc_dict['h_cos1'],
                                   'nc1/h_cos2': nc_dict['h_cos2'],
                                   'nc1/h_cos3': nc_dict['h_cos3']},
                                  step=epoch + 1)
                    val_nc_dict = analysis_feat(self.model, val_targets, val_feats, val_logits, self.args)
                    val_nc.load_dt(val_nc_dict, epoch=epoch + 1, lr=self.optimizer.param_groups[0]['lr'])
                    wandb.log({'nc_val/loss': nc_dict['loss'],
                               'nc_val/acc': nc_dict['acc'],
                               'nc_val/nc1': nc_dict['nc1'],
                               'nc_val/nc2h': nc_dict['nc2_h'],
                               'nc_val/nc2w': nc_dict['nc2_w'],
                               'nc_val/nc3': nc_dict['nc3'],
                               'nc_val/nc3d': nc_dict['nc3_d'],
                               },
                              step=epoch + 1)

                    if (epoch + 1) % (self.args.debug * 5) == 0:
                        fig = plot_nc(nc_dict)
                        wandb.log({"chart": fig}, step=epoch + 1)
                        fig_val = plot_nc(val_nc_dict)
                        wandb.log({"chart_val": fig_val}, step=epoch + 1)

            self.model.train()

        self.log.info('Best Testing Prec@1: {:.3f}\n'.format(best_acc1))
        # Store NC statistics
        filename = os.path.join(self.args.root_model, self.args.store_name, 'train_nc.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(train_nc, f)
        self.log.info('-- Has saved Train NC analysis result to {}'.format(filename))
        filename = os.path.join(self.args.root_model, self.args.store_name, 'val_nc.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(val_nc, f)
        self.log.info('-- Has saved Val NC analysis result to {}'.format(filename))

    def get_feat_logits(self, loader):
        self.model.eval()
        all_logits, all_ncc_logits, all_targets, all_feats = [], [], [], []

        with torch.no_grad():
            for i, (input, target) in enumerate(loader):
                input, target = input.to(self.device), target.to(self.device)
                logit, feat = self.model(input, target, ret='of')  # pred from fc_bc

                if self.knn_classifier is not None:
                    ncc_logit = self.knn_classifier(feat)
                    all_ncc_logits.append(ncc_logit)

                all_logits.append(logit)
                all_targets.append(target)
                all_feats.append(feat)

            all_logits = torch.cat(all_logits)
            all_targets = torch.cat(all_targets)
            all_feats = torch.cat(all_feats)

            if self.knn_classifier is not None:
                all_ncc_logits = torch.cat(all_ncc_logits)
                return all_targets, all_feats, all_logits, all_ncc_logits
            else:
                return all_targets, all_feats, all_logits, []

    def calculate_acc(self, targets, preds):
        eps = np.finfo(np.float64).eps
        cf = confusion_matrix(targets, preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt

        many_shot = self.cls_num_list > 100
        medium_shot = (self.cls_num_list <= 100) & (self.cls_num_list > 20)
        few_shot = self.cls_num_list <= 20

        many_acc = float(sum(cls_acc[many_shot]) * 100 / (sum(many_shot) + eps))
        medium_acc = float(sum(cls_acc[medium_shot]) * 100 / (sum(medium_shot) + eps))
        few_acc = float(sum(cls_acc[few_shot]) * 100 / (sum(few_shot) + eps))

        return cls_acc, many_acc, medium_acc, few_acc

    def SimSiamLoss(self, p, z, version='simplified'):  # negative cosine similarity
        z = z.detach()  # stop gradient

        if version == 'original':
            p = F.normalize(p, dim=1)  # l2-normalize
            z = F.normalize(z, dim=1)  # l2-normalize
            return -(p * z).sum(dim=1).mean()

        elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
            return - F.cosine_similarity(p, z, dim=-1).mean()
        else:
            raise Exception

    def paco_adjust_learning_rate(self, optimizer, epoch, args):
        warmup_epochs = 10
        lr = self.args.lr
        if epoch <= warmup_epochs:
            lr = self.args.lr / warmup_epochs * (epoch + 1)
        else:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs + 1) / (self.args.epochs - warmup_epochs + 1)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_knncentroids(self):
        # print('===> Calculating KNN centroids.')

        torch.cuda.empty_cache()
        self.model.eval()
        feats_all, labels_all = [], []

        # Calculate initial centroids only on training data.
        with torch.set_grad_enabled(False):
            for i, (inputs, labels) in enumerate(self.train_loader):
                if isinstance(inputs, list):
                    inputs = torch.cat(inputs, dim=0)
                    labels = torch.cat((labels, labels), dim=0)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Calculate Features of each training data
                _, feats = self.model(inputs, labels, ret='of')
                feats_all.append(feats.cpu().numpy())
                labels_all.append(labels.cpu().numpy())

        feats = np.concatenate(feats_all)
        labels = np.concatenate(labels_all)
        featmean = feats.mean(axis=0)

        # balanced mean
        if self.args.cbm == 'learn' and self.args.loss == 'arcm' and self.args.bias.lower=='g':
            featmean_cb = self.model.fc.mu.data.cpu().numpy()
        else:  # self.args.cbm == 'compute' and self.args.loss == 'arcm'
            sum_ = torch.zeros((np.max(labels) + 1, feats.shape[-1]))     # [B, d]
            sum_.index_add_(dim=0, index=torch.tensor(labels), source=torch.tensor(feats))   # [K, d]
            cnt_ = torch.bincount(torch.tensor(labels))
            avg_feat = sum_[cnt_ > 0] / cnt_[cnt_ > 0][:, None]  # [K, d]  class-wise mean feat
            featmean_cb = avg_feat.mean(dim=0).numpy()           # class balanced mean

        def get_centroids(feats_, labels_):
            centroids = []
            for i in np.unique(labels_):
                centroids.append(np.mean(feats_[labels_ == i], axis=0))
            return np.stack(centroids)

        # Get unnormalized centorids
        un_centers = get_centroids(feats, labels)

        # Get l2n centorids
        l2n_feats = torch.Tensor(feats.copy())
        norm_l2n = torch.norm(l2n_feats, 2, 1, keepdim=True)
        l2n_feats = l2n_feats / norm_l2n
        l2n_centers = get_centroids(l2n_feats.numpy(), labels)

        # Get cl2n centorids
        cl2n_feats = torch.Tensor(feats.copy())
        cl2n_feats = cl2n_feats - torch.Tensor(featmean)
        norm_cl2n = torch.norm(cl2n_feats, 2, 1, keepdim=True)
        cl2n_feats = cl2n_feats / norm_cl2n
        cl2n_centers = get_centroids(cl2n_feats.numpy(), labels)

        # Get cl2nb centorids
        cl2n_feats = torch.Tensor(feats.copy())
        cl2n_feats = cl2n_feats - torch.Tensor(featmean_cb)
        norm_cl2n = torch.norm(cl2n_feats, 2, 1, keepdim=True)
        cl2n_feats = cl2n_feats / norm_cl2n
        cl2nb_centers = get_centroids(cl2n_feats.numpy(), labels)

        return {'mean': featmean,
                'uncs': un_centers,
                'l2ncs': l2n_centers,
                'cl2ncs': cl2n_centers,
                'cl2nbcs': cl2nb_centers,
                'mean_cb': featmean_cb
                }

    def train(self):
        best_acc1 = 0
        for epoch in range(self.args.start_epoch, self.args.epochs):
            alpha = 1 - (epoch / self.args.epochs) ** 2  # balance loss terms
            losses = AverageMeter('Loss', ':.4e')

            # switch to train mode
            self.model.train()
            start_time = time.time()
            weighted_train_loader = iter(self.weighted_train_loader)

            for i, (inputs, targets) in enumerate(self.train_loader):

                input_org_1 = inputs[0]
                input_org_2 = inputs[1]
                target_org = targets

                try:
                    input_invs, target_invs = next(weighted_train_loader)
                except:
                    weighted_train_loader = iter(self.weighted_train_loader)
                    input_invs, target_invs = next(weighted_train_loader)

                input_invs_1 = input_invs[0][:input_org_1.size()[0]]
                input_invs_2 = input_invs[1][:input_org_2.size()[0]]

                one_hot_org = torch.zeros(target_org.size(0), self.num_classes).scatter_(1, target_org.view(-1, 1), 1)
                one_hot_org_w = self.per_cls_weights.cpu() * one_hot_org
                one_hot_invs = torch.zeros(target_invs.size(0), self.num_classes).scatter_(1, target_invs.view(-1, 1),
                                                                                           1)
                one_hot_invs = one_hot_invs[:one_hot_org.size()[0]]
                one_hot_invs_w = self.per_cls_weights.cpu() * one_hot_invs

                input_org_1 = input_org_1.cuda()
                input_org_2 = input_org_2.cuda()
                input_invs_1 = input_invs_1.cuda()
                input_invs_2 = input_invs_2.cuda()

                one_hot_org = one_hot_org.cuda()
                one_hot_org_w = one_hot_org_w.cuda()
                one_hot_invs = one_hot_invs.cuda()
                one_hot_invs_w = one_hot_invs_w.cuda()

                # Data augmentation
                lam = np.random.beta(self.args.beta, self.args.beta)

                mix_x, cut_x, mixup_y, cutmix_y, mixup_y_w, cutmix_y_w = util.GLMC_mixed(org1=input_org_1,
                                                                                         org2=input_org_2,
                                                                                         invs1=input_invs_1,
                                                                                         invs2=input_invs_2,
                                                                                         label_org=one_hot_org,
                                                                                         label_invs=one_hot_invs,
                                                                                         label_org_w=one_hot_org_w,
                                                                                         label_invs_w=one_hot_invs_w)

                output_1, output_c1, z1, p1 = self.model(mix_x, mixup_y, ret='bc')
                output_2, output_c2, z2, p2 = self.model(cut_x, cutmix_y, ret='bc')
                contrastive_loss = self.SimSiamLoss(p1, z2) + self.SimSiamLoss(p2, z1)

                loss_mix = -torch.mean(torch.sum(F.log_softmax(output_c1, dim=1) * mixup_y, dim=1))
                loss_cut = -torch.mean(torch.sum(F.log_softmax(output_c2, dim=1) * cutmix_y, dim=1))
                loss_mix_w = -torch.mean(torch.sum(F.log_softmax(output_1, dim=1) * mixup_y_w, dim=1))  # class balanced
                loss_cut_w = -torch.mean(
                    torch.sum(F.log_softmax(output_2, dim=1) * cutmix_y_w, dim=1))  # class balanced

                balance_loss = loss_mix + loss_cut
                rebalance_loss = loss_mix_w + loss_cut_w

                loss = alpha * balance_loss + (1 - alpha) * rebalance_loss + self.contrast_weight * contrastive_loss

                losses.update(loss.item(), inputs[0].size(0))

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # ===== finish one epoch
            epoch_time = time.time() - start_time
            self.log.info('====>EPOCH {epoch} Iters {iters}, Epoch Time:{epoch_time:.3f}, Loss:{loss:.4f}.'.format(
                epoch=epoch, iters=len(self.train_loader), epoch_time=epoch_time, loss=losses.avg
            ))
            wandb.log({'train/train_loss': losses.avg,
                       'lr': self.optimizer.param_groups[0]['lr']},
                      step=epoch)

            #  ===== measure NC
            if self.args.debug > 0:
                if (epoch + 1) % self.args.debug == 0:
                    nc_dict = analysis(self.model, self.train_loader, self.args)
                    self.log.info(
                        'Loss:{:.3f}, Acc:{:.2f}, NC1:{:.3f}, NC3:{:.3f}\nWnorm:{}\nHnorm:{}\nWcos:{}\nWHcos:{}'.format(
                            nc_dict['loss'], nc_dict['acc'], nc_dict['nc1'], nc_dict['nc3'],
                            np.array2string(nc_dict['w_norm'], separator=',',
                                            formatter={'float_kind': lambda x: "%.3f" % x}),
                            np.array2string(nc_dict['h_norm'], separator=',',
                                            formatter={'float_kind': lambda x: "%.3f" % x}),
                            np.array2string(nc_dict['w_cos_avg'], separator=',',
                                            formatter={'float_kind': lambda x: "%.3f" % x}),
                            np.array2string(nc_dict['wh_cos'], separator=',',
                                            formatter={'float_kind': lambda x: "%.3f" % x})
                        ))
                    wandb.log({'nc/loss': nc_dict['loss'],
                               'nc/acc': nc_dict['acc'],
                               'nc/nc1': nc_dict['nc1'],
                               'nc/nc2h': nc_dict['nc2_h'],
                               'nc/nc2w': nc_dict['nc2_w'],
                               'nc/nc3': nc_dict['nc3'],
                               'nc/nc3d': nc_dict['nc3_d'],
                               },
                              step=epoch + 1)

                    if self.args.imbalance_type == 'step' and self.args.imbalance_rate < 1.0:
                        wandb.log({'nc1/w_mnorm': nc_dict['w_mnorm'],
                                   'nc1/w_mnorm1': nc_dict['w_mnorm1'],
                                   'nc1/w_mnorm2': nc_dict['w_mnorm2'],
                                   'nc1/h_mnorm': nc_dict['h_mnorm'],
                                   'nc1/h_mnorm1': nc_dict['h_mnorm1'],
                                   'nc1/h_mnorm2': nc_dict['h_mnorm2'],
                                   'nc1/w_cos1': nc_dict['w_cos1'],
                                   'nc1/w_cos2': nc_dict['w_cos2'],
                                   'nc1/w_cos3': nc_dict['w_cos3'],
                                   'nc1/h_cos1': nc_dict['h_cos1'],
                                   'nc1/h_cos2': nc_dict['h_cos2'],
                                   'nc1/h_cos3': nc_dict['h_cos3']},
                                  step=epoch + 1)

                    val_nc_dict = analysis_feat(self.model, val_targets, val_feats, val_logits, self.args)
                    wandb.log({'nc_val/loss': nc_dict['loss'],
                               'nc_val/acc': nc_dict['acc'],
                               'nc_val/nc1': nc_dict['nc1'],
                               'nc_val/nc2h': nc_dict['nc2_h'],
                               'nc_val/nc2w': nc_dict['nc2_w'],
                               'nc_val/nc3': nc_dict['nc3'],
                               'nc_val/nc3d': nc_dict['nc3_d'],
                               },
                              step=epoch + 1)

                    if (epoch + 1) % (self.args.debug * 5) == 0:
                        fig = plot_nc(nc_dict)
                        wandb.log({"chart": fig}, step=epoch + 1)
                        fig_val = plot_nc(val_nc_dict)
                        wandb.log({"chart_val": fig_val}, step=epoch + 1)

            # ============ evaluation ============
            if self.args.imbalance_rate < 1.0:
                cfeats = self.get_knncentroids()
                self.knn_classifier = KNNClassifier(feat_dim=self.model.out_dim,
                                                    num_classes=self.args.num_classes, feat_type='cl2nb',
                                                    dist_type='l2')
                self.knn_classifier.update(cfeats)
            else:
                self.knn_classifier = None

            val_targets, val_feats, val_logits, val_ncc_logits = self.get_feat_logits(self.val_loader)

            # ==== regular validation
            acc1, acc5 = accuracy(val_logits, val_targets, topk=(1, 5))
            cls_acc, many_acc, medium_acc, few_acc = self.calculate_acc(val_targets.cpu().numpy(),
                                                                        val_logits.argmax(1).cpu().numpy())
            self.log.info(
                '---->EPOCH {} Val: Prec@1 {:.3f} Prec@5 {:.3f}, many acc {:.2f}, med acc {:.2f}, few acc {:.2f}'.format(
                    epoch, acc1, acc5, many_acc, medium_acc, few_acc))

            wandb.log({'val/val_acc1': acc1,
                       'val/val_acc5': acc5,
                       'val/val_many': many_acc,
                       'val/val_medium': medium_acc,
                       'val/val_few': few_acc},
                      step=epoch + 1)

            # === validation using Nearest centroid classifier
            if self.args.imbalance_rate < 1.0:
                acc1, acc5 = accuracy(val_ncc_logits, val_targets, topk=(1, 5))
                cls_acc, many_acc, medium_acc, few_acc = self.calculate_acc(val_targets.cpu().numpy(),
                                                                            val_ncc_logits.argmax(1).cpu().numpy())
                self.log.info(
                    '---->EPOCH {} NCC Val: Prec@1 {:.3f} Prec@5 {:.3f}, many acc {:.2f}, med acc {:.2f}, few acc {:.2f}'.format(
                        epoch, acc1, acc5, many_acc, medium_acc, few_acc))

                wandb.log({'knn_val/val_acc1': acc1,
                           'knn_val/val_acc5': acc5,
                           'knn_val/val_many': many_acc,
                           'knn_val/val_medium': medium_acc,
                           'knn_val/val_few': few_acc},
                          step=epoch + 1)

            # === finished validation adjust learning rate
            if self.args.dataset == 'ImageNet-LT' or self.args.dataset == 'iNaturelist2018':
                self.paco_adjust_learning_rate(self.optimizer, epoch, self.args)
            else:
                self.train_scheduler.step()
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
            print(output_best)
            save_checkpoint(self.args, {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_acc1': best_acc1,
            }, is_best, epoch + 1)



