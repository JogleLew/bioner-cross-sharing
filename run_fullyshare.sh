#!/usr/bin/env bash

cuda="0"

mkdir -p tmp

# CUDA_LAUNCH_BLOCKING=1 kernprof -l examples/fullyshared.py
CUDA_VISIBLE_DEVICES=$cuda python examples/fullyshared.py --config config_multitask.py 
