import os
import pprint
import random
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn  as nn
import torch.nn.functional as F


class DotDict(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]


def compute_similarity(x1, x2, metric="Euclidean", normalize=True, centering=True):
    """
    :param x1: input tensor of shape B×P×M .
    :param x2: input tensor of shape B×R×M .
    :param metric: Euclidean or Cosine
    :param normalize: True or False
    :param centering: True or False
    :return: similarity B×P×R
    """
    if metric == "Euclidean":
        # Euclidean distance
        # the cdist dose not support amp
        # distance = torch.cdist(x1, x2)
        if centering:  # centering
            x1 = x1 - x1.mean(1).unsqueeze(1)
            x2 = x2 - x2.mean(1).unsqueeze(1)
        AB = torch.bmm(x1, x2.transpose(1, 2))  # B×P×R
        AA = (x1 * x1).sum(dim=2, keepdim=True)  # B×P×1
        BB = (x2 * x2).sum(dim=2, keepdim=True).reshape(x2.size(0), 1, x2.size(1))  # B×1×R
        # distance = AA.expand_as(AB) - 2 * AB + BB.expand_as(AB)
        distance = AA - 2 * AB + BB

        if normalize:
            distance = distance / x1.size(-1)
        similarity = torch.reciprocal(distance + 1e-8)
        # similarity = -distance
    elif metric == "Cosine":
        x1 = F.normalize(x1, p=2, dim=-1)
        x2 = F.normalize(x2, p=2, dim=-1)
        similarity = torch.bmm(x1, x2.transpose(1, 2))
    elif metric == "Cosine_v2":
        x1 = F.normalize(x1, p=2, dim=-1)
        x2 = F.normalize(x2, p=2, dim=-1)
        similarity = torch.bmm(x1, x2.transpose(1, 2))
        similarity = (similarity + 1) / 2
    else:
        raise NotImplementedError
    return similarity


def summary(model, inputs, configs, device="cuda"):
    # this code is borrowed from https://github.com/sksq96/pytorch-summary
    # we modify to fit our model
    # t_task = configs.t_task // torch.cuda.device_count()

    batch_size = configs.n_way * (configs.k_shot + configs.k_query)

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.CrossEntropyLoss)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(inputs)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(inputs[0][0].size()) * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size
    print("================================================================")
    print('learnable hyper-parameters')
    for name, value in model.named_parameters():
        if not any(x in name for x in ['weight', 'bias']):
            print(name, value.size())
    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary


def cross_entropy(logits, one_hot_targets, reduction='batchmean'):
    logsoftmax_fn = nn.LogSoftmax(dim=1)
    logsoftmax = logsoftmax_fn(logits)
    return - (one_hot_targets * logsoftmax).sum(1).mean()


def compute_acc(logits, one_hot_gts):
    pred = torch.argmax(logits, dim=-1)
    gts = torch.argmax(one_hot_gts, dim=-1)
    correct = (pred == gts).sum()
    acc = correct / float(gts.size(0))
    return acc


def compute_loss_acc(logits, gts, num_class, smoothing=0.1, softmaxed=False):
    #   if smoothing == 0, it's one-hot method
    #   if 0 < smoothing < 1, it's smooth method
    if softmaxed:
        log_prob = torch.log(logits)
    else:
        log_prob = F.log_softmax(logits, dim=1)
    smooth_label = smooth_one_hot(gts, classes=num_class, smoothing=smoothing)
    loss = torch.mean(torch.sum(-smooth_label * log_prob, dim=1))

    pred = torch.argmax(logits, dim=-1)
    correct = (pred == gts).sum()
    acc = correct / float(gts.size(0))

    return loss, acc


def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.1):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist


def str2bool(s):
    s = s.lower()
    true_set = {'yes', 'true', 'gcn_t', 'y', '1'}
    false_set = {'false', 'no', 'f', 'n', '0'}
    if s in true_set:
        return True
    elif s in false_set:
        return False
    else:
        raise ValueError('Excepted {}'.format(' '.join(true_set | false_set)))


def set_seed(seed, local_rank=0):
    if seed == 0:
        torch.backends.cudnn.benchmark = True
    else:
        seed = seed + local_rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def detect_grad_nan(model):
    for param in model.parameters():
        if (param.grad != param.grad).float().sum() != 0:  # nan detected
            param.grad.zero_()


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def set_gpu(gpu):
    gpu_list = [int(x) for x in gpu.split(',')]
    # print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    return gpu_list.__len__()


def remove(file_name):
    try:
        os.remove(file_name)
    except:
        pass


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint([time.strftime('%Y-%m-%d %H:%M:%S'), x])


class Logger(object):
    def __init__(self, log_file):
        self.file = log_file  # file always exists

    def __call__(self, msg, local_rank=-1, init=False, quiet_ter=False, additional_file=None):
        if local_rank in (-1, 0):
            if not quiet_ter:
                pprint(msg)

            if init:
                remove(self.file)
                if additional_file is not None:
                    remove(additional_file)

            with open(self.file, 'a') as log_file:
                log_file.write('%s\n' % msg)
            if additional_file is not None:
                with open(additional_file, 'a') as addition_log:
                    addition_log.write('%s\n' % msg)


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)
        print(f'make path {path}')


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def save_state(net, optimizer=None, scheduler=None, amp_=None):
    save_dict = {
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }

    return save_dict


def run_model(function, world_size, cfg):
    mp.spawn(function, args=(world_size, cfg,), nprocs=world_size, join=True)


def setup(rank, world_size, port=21306):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'{port}'
    # initialize the process group
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def clean_up():
    torch.distributed.destroy_process_group()


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, torch.distributed.ReduceOp.SUM)
    rt /= world_size
    return rt


def get_meta_data(inputs, labels, args, shuffle=False):
    """
    shuffle data with FSL data and local label
    :param inputs: images and global label
    :param labels: local labels
    :param args: args
    :param shuffle: bool, whether shuffle
    :return: images and local label
    """
    images, glo_labels = inputs
    if shuffle:
        # for support set
        sup_img = images[:args.n_way * args.k_shot, ...]
        sup_glo_lab = glo_labels[:args.n_way * args.k_shot]
        sup_loc_lab = labels[:args.n_way * args.k_shot]
        sup_ind = np.random.permutation(args.n_way * args.k_shot)

        sup_img = sup_img[sup_ind]
        sup_glo_lab = sup_glo_lab[sup_ind]
        sup_loc_lab = sup_loc_lab[sup_ind]

        # for query set
        que_img = images[args.n_way * args.k_shot:]
        que_glo_lab = glo_labels[args.n_way * args.k_shot:]
        que_loc_lab = labels[args.n_way * args.k_shot:]
        que_ind = np.random.permutation(args.n_way * args.k_query)

        que_img = que_img[que_ind]
        que_glo_lab = que_glo_lab[que_ind]
        que_loc_lab = que_loc_lab[que_ind]

        return torch.cat([sup_img, que_img], 0), torch.cat([sup_loc_lab, que_loc_lab], 0)
    else:
        return images, labels


def get_similarity(x1, x2, kernel=None, return_distance=False, normalize=False):
    """
    distance with kernel
    :param x1: input tensor of shape P×M
    :param x2: input tensor of shape R×M
    :param kernel: kernel M×M
    :param normalize:
    :param return_distance: whether to return distance
    :return: distance with P×R
    """
    if normalize:
        x1= F.normalize(x1, p=2, dim=-1)
        x2 = F.normalize(x2, p=2, dim=-1)
    if kernel:
        AA = (x1.mm(kernel) * x1).sum(dim=1, keepdim=True)  # P*1
        BB = (x2.mm(kernel) * x2).sum(dim=1).unsqueeze(0)  # 1*R
        AB = x1.mm(kernel).mm(x2.transpose(0, 1))
    else:
        AA = (x1 * x1).sum(dim=1, keepdim=True)  # P*1
        BB = (x2 * x2).sum(dim=1).unsqueeze(0)  # 1*R
        AB = x1.mm(x2.transpose(0, 1))
    # BA = x2.mm(kernel).mm(x1.transpose(0, 1))
    distance = AA - 2 * AB + BB
    distance = distance / x1.size(-1)

    if return_distance:
        return distance
    similarity = torch.reciprocal(distance + 1e-8)
    return similarity


def svd(A, use_cpu, use_double, some=False):
    if use_cpu:
        A = A.cpu()
    if use_double:
        A = A.double()
    U, S, V = torch.svd(A, some=some)
    if use_double:
        U = U.float()
        S = S.float()
        V = V.float()
    if use_cpu:
        U = U.cuda()
        S = S.cuda()
        V = V.cuda()

    return U, S, V


def get_geodesic_distance(x1, x2, p=5):
    """
    geodesic distance
    :param x1: 640 * 25
    :param x2: 640 * 25
    :param p: Grassmannian dim
    :return:
    """
    u1, s1, v1 = svd(x1, use_cpu=False, use_double=True, some=True)
    u2, s2, v2 = svd(x2, use_cpu=False, use_double=True, some=True)
    x1_basis = u1[:, :p]
    x2_basis = u2[:, :p]
    dot = x2_basis.t.mm(x1_basis)  # p * p
    _, s, _ = svd(dot, use_cpu=False, use_double=True, some=True)
    distance1 = (s * s).mean()
    distance2 = ((s1 - s2) ** 2).mean()
    distance = distance2 + distance1
    return distance


def center(A):
    return A - A.mean(0, keepdim=True)


def get_proj(x, dim=5):
    # x: shape: n, out_dim, hw
    u = torch.svd(x, some=False)[0][..., :dim]  # n, out_dim, dim
    # u = svd(x, some=False, use_cpu=False, use_double=False)[0][..., :dim]  # n, out_dim, dim
    proj = u.bmm(u.transpose(-1, -2))  # n, out_dim, out_dim
    return proj
