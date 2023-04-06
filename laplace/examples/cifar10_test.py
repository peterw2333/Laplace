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
model = wrn.WideResNet(16, 4, num_classes=10).cuda().eval()

print([name for name, _ in model.named_modules()])

# util.download_pretrained_model()
model.load_state_dict(torch.load('/code/bnn/Laplace/examples/temp/CIFAR10_plain_uploaded.pt'))
print('test')


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
print(f'[MAP] Time used for prediction: {predict_time}')
acc_map = (probs_map.argmax(-1) == targets).float().mean()
ece_map = ECE(bins=15).measure(probs_map.numpy(), targets.numpy())
nll_map = -dists.Categorical(probs_map).log_prob(targets).mean()

print(f'[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}')

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

# print(f'[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}')

# Laplace
from laplace.utils import LargestVarianceDiagLaplaceSubnetMask, LastLayerSubnetMask, LargestMagnitudeSubnetMask, ModuleNameSubnetMask

masking_mode = "ModuleNameLargestMagnitude"
print(masking_mode)

while True:
    if masking_mode == "LastLayer":
        subnetwork_mask = LastLayerSubnetMask(model)
        subnetwork_indices = subnetwork_mask.select(train_loader)
        print(subnetwork_indices)
        print(f"Number of params: {subnetwork_indices.shape[0]}")

        la = Laplace(model, 'classification',
                    subset_of_weights='last_layer',
                    hessian_structure='full')
    elif masking_mode == "LargestVarianceDiag":
        diag_laplace_model = Laplace(model, 'classification',
                    subset_of_weights='all',
                    hessian_structure='diag')
        subnetwork_mask = LargestVarianceDiagLaplaceSubnetMask(model, n_params_subnet=10, diag_laplace_model=diag_laplace_model)
        subnetwork_indices = subnetwork_mask.select(val_loader)
        print(f"Number of params: {subnetwork_indices.shape[0]}")

        la = Laplace(model, 'classification',
                    subset_of_weights='subnetwork',
                    hessian_structure='full',
                    subnetwork_indices=subnetwork_indices)
    elif masking_mode == "LargestMagnitude":
        subnetwork_mask = LargestMagnitudeSubnetMask(model, n_params_subnet=10)
        subnetwork_indices = subnetwork_mask.select(train_loader)
        print(f"Number of params: {subnetwork_indices.shape[0]}")

        la = Laplace(model, 'classification',
                    subset_of_weights='subnetwork',
                    hessian_structure='full',
                    subnetwork_indices=subnetwork_indices)
    elif masking_mode == "ModuleName":
        module_names = ['fc']
        print(module_names)
        subnetwork_mask = ModuleNameSubnetMask(model, module_names=module_names)
        subnetwork_indices = subnetwork_mask.select(train_loader)
        print(subnetwork_indices)
        print(f"Number of params: {subnetwork_indices.shape[0]}")

        la = Laplace(model, 'classification',
                    subset_of_weights='subnetwork',
                    hessian_structure='full',
                    subnetwork_indices=subnetwork_indices)
    elif masking_mode == "ModuleNameLargestMagnitude":
        module_names = ['block3.layer.1.conv2']
        n_params_subnet = 100
        module_name_mask = ModuleNameSubnetMask(model, module_names=module_names)
        module_name_indices = module_name_mask.select(train_loader)
        largest_magnitude_mask = LargestMagnitudeSubnetMask(model, n_params_subnet=10)
        largest_magnitude_score = largest_magnitude_mask.compute_param_scores(train_loader)

        module_largest_magnitude_score = torch.index_select(largest_magnitude_score, 0, module_name_indices)
        idx = torch.argsort(module_largest_magnitude_score, descending=True)[:n_params_subnet]
        subnetwork_indices = module_name_indices[idx]
        print(f"Number of params: {subnetwork_indices.shape[0]}")

        la = Laplace(model, 'classification',
                    subset_of_weights='subnetwork',
                    hessian_structure='full',
                    subnetwork_indices=subnetwork_indices)

    else:
        mask_correct = False
        masking_mode = input("incorrect masking mode. re-input here:")
        continue
    break

print("complete")

la.fit(train_loader)
la.optimize_prior_precision(method='marglik')
# la.optimize_prior_precision(method='CV', val_loader=val_loader, lr=1e-3)

start_time = time.time()
probs_laplace = predict(test_loader, la, laplace=True)
predict_time = time.time() - start_time
print(f'[Laplace] Time used for prediction: {predict_time}')
acc_laplace = (probs_laplace.argmax(-1) == targets).float().mean()
ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean()

print(f'[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}')




