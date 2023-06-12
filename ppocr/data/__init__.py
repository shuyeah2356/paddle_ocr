# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import numpy as np
import skimage
import paddle
import signal
import random

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import copy
from paddle.io import Dataset, DataLoader, BatchSampler, DistributedBatchSampler
import paddle.distributed as dist

from ppocr.data.imaug import transform, create_operators
from ppocr.data.simple_dataset import SimpleDataSet
from ppocr.data.lmdb_dataset import LMDBDataSet, LMDBDataSetSR
from ppocr.data.pgnet_dataset import PGDataSet
from ppocr.data.pubtab_dataset import PubTabDataSet

__all__ = ['build_dataloader', 'transform', 'create_operators']


def term_mp(sig_num, frame):
    """ kill all child processes
    """
    pid = os.getpid()
    pgid = os.getpgid(os.getpid())
    print("main proc {} exit, kill process group " "{}".format(pid, pgid))
    os.killpg(pgid, signal.SIGKILL)


def build_dataloader(config, mode, device, logger, seed=None):
    """dataloader:
    选择支持的数据集名称，从配置文件中获取数据集名称，判别数据集的有效性
    从配置文件中读取模式：train/eval/test
    从配置文件中读取数据加载的参数：batch_size_per_card、drop_last、shuffle、num_workers、use_shared_memory
    如果所选择的模型为train,采用多张卡分布式训练；test/eval模式则采用单卡运行""" 
    config = copy.deepcopy(config)

    support_dict = [
        'SimpleDataSet', 'LMDBDataSet', 'PGDataSet', 'PubTabDataSet',
        'LMDBDataSetSR'
    ]
    module_name = config[mode]['dataset']['name']
    assert module_name in support_dict, Exception(
        'DataSet only support {}'.format(support_dict))
    assert mode in ['Train', 'Eval', 'Test'
                    ], "Mode should be Train, Eval or Test."

    dataset = eval(module_name)(config, mode, logger, seed)
    loader_config = config[mode]['loader']
    batch_size = loader_config['batch_size_per_card']
    drop_last = loader_config['drop_last']
    shuffle = loader_config['shuffle']
    num_workers = loader_config['num_workers']
    if 'use_shared_memory' in loader_config.keys():
        use_shared_memory = loader_config['use_shared_memory']
    else:
        use_shared_memory = True

    if mode == "Train":
        # Distribute data to multiple cards
        batch_sampler = DistributedBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last)
    else:
        # Distribute data to single card
        batch_sampler = BatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last)
    # 用于自定义样本的批处理方式。collate_fn函数将数据集中的一批样本组合成一个批次，以便在训练中进行处理
    if 'collate_fn' in loader_config:
        from . import collate_fn
        collate_fn = getattr(collate_fn, loader_config['collate_fn'])()
    else:
        collate_fn = None
    # dataloader
    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        places=device,
        num_workers=num_workers,
        return_list=True,
        use_shared_memory=use_shared_memory,
        collate_fn=collate_fn)

    # support exit using ctrl+c
    signal.signal(signal.SIGINT, term_mp)
    signal.signal(signal.SIGTERM, term_mp)

    return data_loader
