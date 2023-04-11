import warnings
warnings.simplefilter("ignore", UserWarning)

import torch
import torch.distributions as dists
import torch.nn as nn
import numpy as np
import helper.wideresnet as wrn
import helper.dataloaders as dl
from helper import util
from netcal.metrics import ECE

from laplace import Laplace
import torchvision
import torchvision.transforms as T

from tqdm import tqdm
from models import MLP, LeNet5
import time


np.random.seed(7777)
torch.manual_seed(7777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_size = 784 # 28x28
hidden_size = 500
num_classes = 10
num_epochs = 3
batch_size = 100
learning_rate = 0.001

device = torch.device('cuda')

# MLP
# transform=T.Compose([T.ToTensor(), T.Lambda(lambda x: torch.flatten(x))])
# model = MLP(hidden_size=100).to(device)

# LeNet5
transform=T.ToTensor()
model = LeNet5().to(device)

temp_dataset = torchvision.datasets.MNIST(root='./temp',
                                            train=True,
                                            transform=transform,
                                            download=True)

train_dataset, val_dataset = torch.utils.data.random_split(temp_dataset, [55000, 5000])

test_dataset = torchvision.datasets.MNIST(root='./temp',
                                            train=False,
                                            transform=transform,
                                            download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                         shuffle=True)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                         shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

num_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in tqdm(enumerate(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step[{i+1}/{num_steps}], Loss: {loss.item():.4f}')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

acc = 100.0 * n_correct / n_samples
print(f'Accuracy of the network on the 10000 test images: {acc} %')

@torch.no_grad()
def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x.cuda()))
        else:
            py.append(torch.softmax(model(x.cuda()), dim=-1))

    return torch.cat(py).cpu()

targets = torch.cat([y for x, y in test_loader], dim=0).cpu()
start_time = time.time()
probs_map = predict(test_loader, model, laplace=False)
predict_time = time.time() - start_time
print(f'[MAP] Time used for prediction: {predict_time}')
acc_map = (probs_map.argmax(-1) == targets).float().mean()
ece_map = ECE(bins=15).measure(probs_map.numpy(), targets.numpy())
nll_map = -dists.Categorical(probs_map).log_prob(targets).mean()

print(f'[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}')

# Laplace
from laplace.utils import LargestVarianceDiagLaplaceSubnetMask, LastLayerSubnetMask, LargestMagnitudeSubnetMask, ModuleNameSubnetMask

masking_mode = "ModuleName"
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
        module_names = ['conv1']
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
        module_names = ['conv1']
        print(module_names)
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

    elif masking_mode == "ModuleNameLargestVariance":
        module_names = ['block1.layer.0.conv1','block1.layer.0.conv2']
        print(module_names)
        n_params_subnet = 10000

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
        print(f"Number of params: {subnetwork_indices.shape[0]}")

        la = Laplace(model, 'classification',
                    subset_of_weights='subnetwork',
                    hessian_structure='full',
                    subnetwork_indices=subnetwork_indices)
    elif masking_mode == "ConvLayerLargestMagnitude":
        module_name = "block1.layer.0.conv1"
        k = 64
        print(module_name)
        module_names = [module_name]
        module_name_mask = ModuleNameSubnetMask(model, module_names=module_names)
        module_name_indices = module_name_mask.select(train_loader)

        weights_name = module_name + ".weight"

        print(weights_name)
        for name, params in model.named_parameters():
            if name == weights_name:
                target_params = params
                break

        target_params = torch.abs(target_params)
        target_params = target_params.flatten(start_dim=2) # 64 * 16 * 3 * 3 -> 64 * 16 * 9
        max_indices = torch.zeros_like(target_params)
        for idx in range(target_params.shape[0]):

            topk = torch.topk(torch.flatten(target_params[idx,:,:]), k=k).values
            for value in list(topk):
                max_indices[idx,:,:] = torch.logical_or(max_indices[idx,:,:], (target_params[idx,:,:] == value))

        torch.set_printoptions(threshold=10000)
        max_indices = max_indices.view(-1)
        subnetwork_indices = torch.flatten(torch.nonzero(max_indices)) + module_name_indices[0]

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
