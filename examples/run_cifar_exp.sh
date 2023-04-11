CUDA_VISIBLE_DEVICES=0
python cifar10_test.py \
    --masking_mode ModuleNameLargestMagnitude \
    --n_params_subnet 500 \
    --module_names layer1.0.conv1
