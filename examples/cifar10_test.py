import warnings
warnings.simplefilter("ignore", UserWarning)

import torch
import torch.distributions as dists
import numpy as np
import helper.wideresnet as wrn
import helper.dataloaders as dl
from helper import util
from netcal.metrics import ECE
import torch.nn as nn

from laplace import Laplace

import time


import torch
import torch.hub as hub

from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--masking_mode', type=str, required=True)
parser.add_argument('--n_params_subnet', type=int, required=False, default=-1)
parser.add_argument('--module_names', nargs='+', required=False, default=None)
args = parser.parse_args()

import logging

# create logger
logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)

# create file handler and set level to INFO
file_handler = logging.FileHandler('cifar10_exp.log')
file_handler.setLevel(logging.INFO)

# create console handler and set level to DEBUG
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to handlers
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def patch_hub_with_proxy():
    download_url_to_file = hub.download_url_to_file

    def _proxy_download_url_to_file(
        url: str,
        *args,
        **kwargs
    ):
        if url.startswith("https://github.com"):
            cdn_url = "https://ghproxy.com/" + url
            return download_url_to_file(cdn_url, *args, **kwargs)
    hub.download_url_to_file = _proxy_download_url_to_file

    def _git_archive_link(repo_owner, repo_name, ref):
        return f"https://github.com/{repo_owner}/{repo_name}/archive/refs/heads/{ref}.zip"
    hub._git_archive_link = _git_archive_link


patch_hub_with_proxy()

np.random.seed(7777)
torch.manual_seed(7777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

temp_dataset = dl.CIFAR10_dataset(train=True)
train_dataset, val_dataset = torch.utils.data.random_split(temp_dataset, [49000, 1000])

train_loader = torch.utils.data.DataLoader(temp_dataset, batch_size=128,
                                         shuffle=True, num_workers=4)

val_loader = torch.utils.data.DataLoader(temp_dataset, batch_size=128,
                                         shuffle=False, num_workers=4)


test_loader = dl.CIFAR10(train=False)


targets = torch.cat([y for x, y in test_loader], dim=0).cpu()

# The model is a standard WideResNet 16-4
# Taken as is from https://github.com/hendrycks/outlier-exposure
# model = wrn.WideResNet(16, 4, num_classes=10).cuda().eval()
# # util.download_pretrained_model()
# model.load_state_dict(torch.load('/code/bnn/Laplace/examples/temp/CIFAR10_plain_uploaded.pt'))

model_name = "cifar10_resnet20"
logger.info(f"MODEL_NAME: {model_name}")
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", force_reload=True, pretrained=True)

device = torch.device("cuda")
model = model.to(device)

logger.info([name for name, _ in model.named_modules()])



@torch.no_grad()
def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x.cuda()))
        else:
            py.append(torch.softmax(model(x.cuda()), dim=-1))

    return torch.cat(py).cpu()

start_time = time.time()
probs_map = predict(test_loader, model, laplace=False)
predict_time = time.time() - start_time
logger.info(f'[MAP] Time used for prediction: {predict_time}')
acc_map = (probs_map.argmax(-1) == targets).float().mean()
ece_map = ECE(bins=15).measure(probs_map.numpy(), targets.numpy())
nll_map = -dists.Categorical(probs_map).log_prob(targets).mean()

logger.info(f'[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}')

# Laplace
# la = Laplace(model, 'classification',
#              subset_of_weights='last_layer',
#              hessian_structure='kron')

# from laplace.utils import LargestMagnitudeSubnetMask
# subnetwork_mask = LargestMagnitudeSubnetMask(model, n_params_subnet=100)
# subnetwork_indices = subnetwork_mask.select()

# la = Laplace(model, 'classification',
#              subset_of_weights='subnetwork',
#              hessian_structure='diag',
#              subnetwork_indices=subnetwork_indices)
# la.fit(train_loader)
# la.optimize_prior_precision(method='marglik')

# probs_laplace = predict(test_loader, la, laplace=True)
# acc_laplace = (probs_laplace.argmax(-1) == targets).float().mean()
# ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
# nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean()

# logger.info(f'[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}')


# Laplace
from laplace.utils import LargestVarianceDiagLaplaceSubnetMask, LastLayerSubnetMask, LargestMagnitudeSubnetMask, ModuleNameSubnetMask

masking_mode = args.masking_mode
module_names = args.module_names
n_params_subnet = args.n_params_subnet
logger.info(f"MASKING MODE: {masking_mode}")
logger.info(f"MODULE NAMES: {module_names}")
logger.info(f"SUBNET PARAMETER COUNT: {n_params_subnet}")

seeds = [0, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]

acc_laplace = []
ece_laplace = []
nll_laplace = []

for i in tqdm(range(len(seeds))):
    np.random.seed(seeds[i])
    torch.manual_seed(seeds[i])
    if masking_mode == "LastLayer":
        subnetwork_mask = LastLayerSubnetMask(model)
        subnetwork_indices = subnetwork_mask.select(train_loader)
        logger.info(subnetwork_indices)
        logger.info(f"Number of params: {subnetwork_indices.shape[0]}")

        la = Laplace(model, 'classification',
                    subset_of_weights='last_layer',
                    hessian_structure='full')
    elif masking_mode == "LargestVarianceDiag":
        diag_laplace_model = Laplace(model, 'classification',
                    subset_of_weights='all',
                    hessian_structure='diag')
        subnetwork_mask = LargestVarianceDiagLaplaceSubnetMask(model, n_params_subnet=10, diag_laplace_model=diag_laplace_model)
        subnetwork_indices = subnetwork_mask.select(val_loader)
        logger.info(f"Number of params: {subnetwork_indices.shape[0]}")

        la = Laplace(model, 'classification',
                    subset_of_weights='subnetwork',
                    hessian_structure='full',
                    subnetwork_indices=subnetwork_indices)
    elif masking_mode == "LargestMagnitude":
        # n_params_subnet=1000
        subnetwork_mask = LargestMagnitudeSubnetMask(model, n_params_subnet=n_params_subnet)
        subnetwork_indices = subnetwork_mask.select(train_loader)
        logger.info(f"Number of params: {subnetwork_indices.shape[0]}")

        la = Laplace(model, 'classification',
                    subset_of_weights='subnetwork',
                    hessian_structure='full',
                    subnetwork_indices=subnetwork_indices)
    elif masking_mode == "ModuleName":
        # module_names = ['conv1','fc']
        # logger.info(module_names)
        subnetwork_mask = ModuleNameSubnetMask(model, module_names=module_names)
        subnetwork_indices = subnetwork_mask.select(train_loader)
        logger.info(subnetwork_indices)
        logger.info(f"Number of params: {subnetwork_indices.shape[0]}")

        la = Laplace(model, 'classification',
                    subset_of_weights='subnetwork',
                    hessian_structure='full',
                    subnetwork_indices=subnetwork_indices)
    elif masking_mode == "ModuleNameLargestMagnitude":
        # module_names = ['fc']
        # logger.info(module_names)
        # n_params_subnet = 100
        module_name_mask = ModuleNameSubnetMask(model, module_names=module_names)
        module_name_indices = module_name_mask.select(train_loader)
        largest_magnitude_mask = LargestMagnitudeSubnetMask(model, n_params_subnet=10)
        largest_magnitude_score = largest_magnitude_mask.compute_param_scores(train_loader)

        module_largest_magnitude_score = torch.index_select(largest_magnitude_score, 0, module_name_indices)
        idx = torch.argsort(module_largest_magnitude_score, descending=True)[:n_params_subnet]
        subnetwork_indices = module_name_indices[idx]
        logger.info(f"Number of params: {subnetwork_indices.shape[0]}")

        la = Laplace(model, 'classification',
                    subset_of_weights='subnetwork',
                    hessian_structure='full',
                    subnetwork_indices=subnetwork_indices)

    elif masking_mode == "ModuleNameLargestVariance":
        # module_names = ['block1.layer.0.conv1','block1.layer.0.conv2']
        # logger.info(module_names)
        # n_params_subnet = 10000

        module_name_mask = ModuleNameSubnetMask(model, module_names=module_names)
        module_name_indices = module_name_mask.select(train_loader)

        diag_laplace_model = Laplace(model, 'classification',
                    subset_of_weights='all',
                    hessian_structure='diag')
        largest_variance_mask = LargestVarianceDiagLaplaceSubnetMask(model, n_params_subnet=n_params_subnet, diag_laplace_model=diag_laplace_model)
        largest_variance_score = largest_variance_mask.compute_param_scores(train_loader)

        module_largest_variance_score = torch.index_select(largest_variance_score, 0, module_name_indices)
        idx = torch.argsort(module_largest_variance_score, descending=True)[:n_params_subnet]
        subnetwork_indices = module_name_indices[idx]
        logger.info(f"Number of params: {subnetwork_indices.shape[0]}")

        la = Laplace(model, 'classification',
                    subset_of_weights='subnetwork',
                    hessian_structure='full',
                    subnetwork_indices=subnetwork_indices)
    elif masking_mode == "ConvLayerLargestMagnitude":
        logger.info("WARNING: Parameters as arguments are not currently supported. Please hardcode parameters instead.")
        module_names = ["layer1.0.conv1"]
        k = 32
        logger.info(module_names)
        module_name_mask = ModuleNameSubnetMask(model, module_names=module_names)
        module_name_indices = module_name_mask.select(train_loader)

        weights_name = module_name + ".weight"

        logger.info(weights_name)
        for name, params in model.named_parameters():
            if name == weights_name:
                target_params = params
                break
        
        target_params = torch.abs(target_params)
        target_params = target_params.flatten(start_dim=2) # 64 * 16 * 3 * 3 -> 64 * 16 * 9
        max_indices = torch.zeros_like(target_params)
        for idx in range(target_params.shape[1]):

            topk = torch.topk(torch.flatten(target_params[:,idx,:]), k=k).values
            for value in list(topk):
                max_indices[:,idx,:] = torch.logical_or(max_indices[:,idx,:], (target_params[:,idx,:] == value))

        torch.set_logger.infooptions(threshold=10000)
        max_indices = max_indices.view(-1)
        subnetwork_indices = torch.flatten(torch.nonzero(max_indices)) + module_name_indices[0]
        
        logger.info(f"Number of params: {subnetwork_indices.shape[0]}")

        la = Laplace(model, 'classification',
                    subset_of_weights='subnetwork',
                    hessian_structure='full',
                    subnetwork_indices=subnetwork_indices)
    else:
        raise ValueError("Incorrect masking mode.")

    la.fit(train_loader)
    la.optimize_prior_precision(method='marglik')
    # la.optimize_prior_precision(method='CV', val_loader=val_loader, lr=1e-3)

    start_time = time.time()
    probs_laplace = predict(test_loader, la, laplace=True)
    predict_time = time.time() - start_time
    logger.info(f'[Laplace] Time used for prediction: {predict_time}')
    acc = (probs_laplace.argmax(-1) == targets).float().mean()
    ece = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
    nll = -dists.Categorical(probs_laplace).log_prob(targets).mean()

    acc_laplace.append(float(acc))
    ece_laplace.append(float(ece))
    nll_laplace.append(float(nll))

    logger.info(f'[Laplace] Acc.: {acc:.2%}; ECE: {ece:.2%}; NLL: {nll:.4}')

logger.info(f"[Laplace] mean acc={np.average(acc_laplace)}, std={np.std(acc_laplace)}")
logger.info(f"[Laplace] mean ece={np.average(ece_laplace)}, std={np.std(ece_laplace)}")
logger.info(f"[Laplace] mean nll={np.average(nll_laplace)}, std={np.std(nll_laplace)}")

logger.info(f"[Laplace] raw acc:{acc_laplace}")
logger.info(f"[Laplace] raw ece:{ece_laplace}")
logger.info(f"[Laplace] raw nll:{nll_laplace}")





