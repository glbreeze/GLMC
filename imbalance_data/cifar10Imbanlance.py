import torchvision
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

class Cifar10Imbanlance(Dataset):
    def __init__(self, imbalance_rate, num_cls=10, file_path="data/",
                 train=True, transform=None, label_align=True, ):
        self.transform = transform
        self.label_align = label_align
        assert 0.0 < imbalance_rate <= 1, "imbanlance_rate must 0.0 < imbanlance_rate <= 1"
        self.imbalance_rate = imbalance_rate

        self.num_cls = num_cls
        self.data = self.produce_imbalance_data(file_path=file_path, train=train,imbalance_rate=self.imbalance_rate)
        self.x = self.data['x']
        self.targets = self.data['y'].tolist()
        self.y = self.data['y'].tolist()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x, y = self.x[item], self.y[item]
        x = Image.fromarray(x)
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def get_per_class_num(self):
        return self.class_list

    def produce_imbalance_data(self, imbalance_rate, file_path="/data", train=True):

        train_data = torchvision.datasets.CIFAR10(
            root=file_path,
            train=train,
            download=True,
        )
        x_train = train_data.data
        y_train = train_data.targets
        y_train = np.array(y_train)

        data_percent = []
        data_num = int(x_train.shape[0] / self.num_cls)

        for cls_idx in range(self.num_cls):
            if train:
                num = data_num * (imbalance_rate ** (cls_idx / (self.num_cls - 1)))
                data_percent.append(int(num))
            else:
                num = data_num
                data_percent.append(int(num))
        self.class_list = data_percent
        if train:
            print("imbalance_ration is {}".format(data_percent[0] / data_percent[-1]))
            print("per class num: {}".format(data_percent))


        rehearsal_data = None
        rehearsal_label = None
        for i in range(1, self.num_cls + 1):
            index = (y_train >= i - 1) & (y_train < i)
            task_train_x = x_train[index]
            label = y_train[index]

            data_num = task_train_x.shape[0]
            index = np.random.choice(data_num, data_percent[i - 1],replace=False)
            tem_data = task_train_x[index]
            tem_label = label[index]
            if rehearsal_data is None:
                rehearsal_data = tem_data
                rehearsal_label = tem_label
            else:
                rehearsal_data = np.concatenate([rehearsal_data, tem_data], axis=0)
                rehearsal_label = np.concatenate([rehearsal_label, tem_label], axis=0)

        task_split = {
            "x": rehearsal_data,
            "y": rehearsal_label,
        }
        return task_split
