import numpy as np
import torch.nn as nn
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device
from utils.conf import checkpoint_path
from utils.util import create_if_not_exists
import os


class FederatedModel(nn.Module):
    """
    Federated learning model.
    """
    NAME = None
    N_CLASS = None

    def __init__(self, nets_list: list,
                 args: Namespace, transform: torchvision.transforms) -> None:
        super(FederatedModel, self).__init__()
        self.nets_list = nets_list
        self.args = args
        self.transform = transform

        self.num_samples = []

        # For Online
        self.random_state = np.random.RandomState()
        self.online_num = np.ceil(self.args.parti_num * self.args.online_ratio).item()
        self.online_num = int(self.online_num)

        self.global_net = None
        self.device = get_device(device_id=self.args.device_id)

        self.local_epoch = args.local_epoch
        self.local_lr = args.local_lr
        self.trainloaders = None
        self.testlodaers = None

        self.epoch_index = 0 # Save the Communication Index

        self.checkpoint_path = checkpoint_path() + self.args.dataset + '/' + self.args.structure + '/'
        create_if_not_exists(self.checkpoint_path)
        self.net_to_device()

    def net_to_device(self):
        for net in self.nets_list:
            net.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_scheduler(self):
        return

    def ini(self):
        pass

    def col_update(self, communication_idx, publoader):
        pass

    def loc_update(self, priloader_list):
        pass

    def load_pretrained_nets(self):
        if self.load:
            for j in range(self.args.parti_num):
                pretrain_path = os.path.join(self.checkpoint_path, 'pretrain')
                save_path = os.path.join(pretrain_path, str(j) + '.ckpt')
                self.nets_list[j].load_state_dict(torch.load(save_path, self.device))
        else:
            pass

    def copy_nets2_prevnets(self):
        nets_list = self.nets_list
        prev_nets_list = self.prev_nets_list
        for net_id, net in enumerate(nets_list):
            net_para = net.state_dict()
            prev_net = prev_nets_list[net_id]
            prev_net.load_state_dict(net_para)

    def aggregate_nets(self, freq=None):
        global_net = self.global_net
        nets_list = self.nets_list

        c_samples_list = self.num_samples

        online_clients = self.online_clients
        global_w = self.global_net.state_dict()

        if self.args.averaing == 'weight':
            online_clients_dl = [self.trainloaders[online_clients_index] for online_clients_index in online_clients]
            online_clients_len = [dl.sampler.indices.size for dl in online_clients_dl]
            online_clients_all = np.sum(online_clients_len)
            freq = online_clients_len / online_clients_all

            '''
            # domain-aware
            if self.args.dataset == 'fl_pacs':
                C_NUM = 7
            else:
                C_NUM = 10

            ALL_SAMPLES = np.sum(c_samples_list)
            g_arr = np.full((C_NUM,), 0.25)
            # print(g_arr)

            c_dis_list = []
            c_num_list = []

            for c_sam_num in c_samples_list:
                c_arr = np.full((C_NUM,), c_sam_num/ALL_SAMPLES)
                # print(c_arr)
                c_dis = np.sqrt(0.5 * np.sum((c_arr - g_arr) ** 2))
                # print(c_dis)
                # c_dis = np.sqrt(0.5 * np.sum((c_arr - g_arr) ** 2))
                c_dis_list.append(c_dis)
                c_num_list.append(c_sam_num/ALL_SAMPLES)
            weight_list = []
            # print('-----------------------------------')
            for c_value, d_value in zip(c_num_list, c_dis_list):
                # print(c_value, d_value)
                all_v = self.args.agg_a * c_value - self.args.agg_b * d_value
                # print(all_v)
                # sigmoid_v = max(0, all_v)
                sigmoid_v = 1 / (1 + np.exp(-all_v))
                # print(sigmoid_v)
                weight_list.append(sigmoid_v)

            # print('------------------------')
            # print(weight_list)
            all_w_value = np.sum(weight_list)
            freq = []
            for c_w in weight_list:
                freq.append(c_w/all_w_value)
            # print(freq)
            '''

        else:
        # if freq == None:
            parti_num = len(online_clients)
            freq = [1 / parti_num for _ in range(parti_num)]

        first = True
        for index,net_id in enumerate(online_clients):
            net = nets_list[net_id]
            net_para = net.state_dict()

            # if net_id == 0:
            if first:
                first = False
                for key in net_para:
                    global_w[key] = net_para[key] * freq[index]
            else:
                for key in net_para:
                    global_w[key] += net_para[key] * freq[index]

        global_net.load_state_dict(global_w)


        for _, net in enumerate(nets_list):
            net.load_state_dict(global_net.state_dict())

            # global_dict = global_net.state_dict()
            # net_dict = net.state_dict()
            # skip_keys = ['separation', 'recalibration', 'aux']
            # new_dict = global_dict.copy()
            # for k, v in global_dict.items():
            #     if any(sk in k for sk in skip_keys):
            #         new_dict[k] = net_dict[k]
            # net.load_state_dict(new_dict)