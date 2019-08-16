import os
import random as rnd
rnd.seed(123)
import codecs
import more_itertools
import numpy as np
from sklearn.model_selection import ParameterGrid
import six
import shutil
import subprocess

cmd_env = """
source activate allennlp

"""
cmd_template = """python3 examples/multitask.py --config config_multitask.py --grid "%s" 
"""
gpus = [0]
param_grid = {
        'config.multitask.adv_loss_coef': [0, 0.001, 0.01, 0.1],
        'config.multitask.diff_loss_coef': [0, 0.001, 0.01, 0.1]
}


def prepare_grid_dir():
    if os.path.exists('grid'):
        shutil.rmtree('grid')
    os.mkdir('grid')


def generate_grid_search_shell():
    cmd_list = []
    for setting in ParameterGrid(param_grid):
        cmd_list.append(cmd_template % setting)
    print('#cmd:', len(cmd_list))

    # shuffle and divide cmds to different gpus
    rnd.shuffle(cmd_list)
    cmd_bucket_list = more_itertools.divide(len(gpus), cmd_list)
    for i, gpu_id in enumerate(gpus):
        with codecs.open('grid/grid.%s.sh' % (gpu_id,), 'w', 'utf-8') as f_out:
            f_out.write(cmd_env)
            f_out.write('\n'.join(map(lambda x: 'CUDA_VISIBLE_DEVICES=%d %s' %
                                      (gpu_id, x), cmd_bucket_list[i])) + '\n')


def execute_script():
    grid_scripts = os.listdir('grid')
    sub_processes = []
    for grid_script in grid_scripts:
        child = subprocess.Popen(['bash', os.path.join('grid', grid_script)])
        sub_processes.append(child)
    for proc in sub_processes:
        proc.wait()


prepare_grid_dir()
generate_grid_search_shell()
execute_script()
