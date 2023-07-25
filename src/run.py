# %%
# 初始化
import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List
import click

# %%
# 設定參數
DATASET = click.prompt(
    "Input path to dataset",
    default="/home/shengfeng/Documents/thesis_works/data/genomeAndGP/",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    value_proc=Path,
)

# %%
# 準備 MsDroid 用 APK 路徑
def createSymbolicApkPath(dataset: Path) -> Dict[str, Path]:
    datasetName = dataset.stem

    softLinkFolder = dataset / "msDroid"
    softLinkFolder.mkdir(exist_ok=True)

    softLinkToAPKDict = {
    }

    for apkType in ["benign", "malicious"]:
        realLinkToAPKs = dataset / "benign" / "apks"
        assert realLinkToAPKs.exists()

        folderName = f"{datasetName}_{apkType}"

        softLinkToAPKs = softLinkFolder / folderName

        if not softLinkToAPKs.exists():
            os.symlink(realLinkToAPKs, softLinkToAPKs, target_is_directory=True)

        softLinkToAPKDict[folderName] = {
            "label": "1" if apkType == "malicious" else "0",
            "path": softLinkToAPKs
        }

    return softLinkToAPKDict

pathToAPKs = createSymbolicApkPath(DATASET)

# %%
# 載入 run.py
# 幹你娘 %load 沒有用

import sys
import os
import logging
from training import set_train_config, GraphDroid
from graph import generate_graph
from utils import makedirs, set_logger
from main import generate_behavior_subgraph
exp_base = './training/Experiments'
graph_base = f'./training/Graphs'
logger = set_logger(logging.getLogger())

def start(logger, args):
    input_dir = args.input
    assert os.path.exists(input_dir), f"{input_dir} is not exists"
    apk_base = os.path.abspath(os.path.join(input_dir,'../'))
    db_name = input_dir.split(apk_base)[-1].strip('/')
    
    output_dir = args.output
    makedirs(output_dir)
    label = float(args.label)
    dbs = ["torrent_benign", "torrent_malicious"]
    tpl = args.tpl
    hop = args.hop
    batch_size = args.batch_size
    train_rate = args.train_rate
    norm_opcode = args.norm_op
    mask = args.mask
    global_pool = args.global_pool
    dimension = args.dimension
    lossfunc = args.lossfunc
    dev = args.dev
    epoch = args.epoch
    force = args.force
    continue_train = args.continue_train
    testonly = args.testonly
    
    exp_dir = f'./training/Graphs/{db_name}/HOP_{hop}/TPL_{tpl}'
    if not os.path.exists(f'{exp_dir}/dataset.pt'):
        makedirs('Mappings')
        import time
        T1 = time.process_time()    
        '''
        ./training/Graphs/<db_name>/processed/data_<apk_id>_<subgraph_id>.pt
        '''
        num_apk = generate_behavior_subgraph(apk_base, db_name, output_dir, args.deepth, label, hop=hop, tpl=tpl, training=True, api_map=True)
        T2 = time.process_time()
        print(f'Generate Behavior Subgraphs for {num_apk} APKs: {T2-T1}')
        testonly = True if num_apk==1 else False

    model_config = set_train_config(batch_size=batch_size, train_rate=train_rate, global_pool=global_pool, lossfunc=lossfunc, dimension=dimension)
    graph_droid = GraphDroid(hop, tpl, dbs, norm_opcode=norm_opcode, mask=mask, model_config=model_config, exp_base=args.exp_base, graph_base=args.graph_base, logger=logger)
    
    graph_droid.train_and_test(epoch, force=force, continue_train=continue_train, dev=dev, testonly=testonly)

def parse_args(args=None):
    import argparse

    parser = argparse.ArgumentParser(description='MsDroid Trainer.')
    # Generate behavior subgraphs
    parser.add_argument('--input', '-i', help='APK directory')
    parser.add_argument('--output', '-o', help='output directory', default=f'{sys.path[0]}/Output')
    parser.add_argument('--device', '-d', help='device for model test', default='cpu')
    parser.add_argument('--batch', '-b', help='batch size for model test', default=16)
    parser.add_argument('--label', '-l', help='dataset label: malware(1) / benign(0), unnecessary if only prediction needed.', default=1)
    parser.add_argument('--deepth', '-dp', help='deepth of tpl seaching', default=3)
    # Training
    parser.add_argument('--dbs', action="append", help='Datasets to train.')
    parser.add_argument('--tpl', type=bool, default=True, help='TPL simplified subgraphs.')
    parser.add_argument('--hop', type=int, default=2, help='K-hop based subgraphs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for Dataloader.')
    parser.add_argument('--train_rate', type=float, default=0.8, help='Training rate.')
    parser.add_argument('--norm_op', type=bool, default=False, help='Normalize opcodes feature.')
    parser.add_argument('--mask', type=int, default=-1, help='Mask node features. 0: disable opcodes, 1: disable permission, 2: disable both')
    parser.add_argument('--global_pool', type=str, default='mix', help='Global pooling method for graph classification.')
    parser.add_argument('--lossfunc', type=int, default=0, help='Index of loss function.')
    parser.add_argument('--dimension', type=int, default=128, help='Hidden layer graph embedding dimension.')
    parser.add_argument('--dev', type=int, default=0, help='GPU device id.')
    parser.add_argument('--exp_base', type=str, default=exp_base, help='Dir to put exp results.')
    parser.add_argument('--graph_base', type=str, default=graph_base, help='Dir for graphs.')
    # For Train (`train_and_test`)
    parser.add_argument('--epoch', type=int, default=1, help='Training epoches.')
    parser.add_argument('--force', type=bool, default=False, help='Force new train in exp_base with same config.')
    parser.add_argument('--continue_train', type=bool, default=False, help='Continue to train from last checkpoint.')
    parser.add_argument('--testonly', type=bool, default=False, help='!!!!!!')
    args = parser.parse_args(args)
    return args

# %%
# 跑資料前處理

for folderName, pack in pathToAPKs.items():
    label, pathToApk = pack["label"], pack["path"]

    params = [
            "-i",
            str(pathToApk),
            "--label",
            label,
            "-o",
            f"/home/shengfeng/Documents/MsDroid/{folderName}",
            "--force",
            "True",
            "--testonly",
            "False",
            "--device",
            "cuda",
            "--epoch",
            "1",
        ]

    print(f"{params=}")

    parser = parse_args(
        params
    )
    
    # print(parser.dbs)
    start(logger, parser)

# %%
# 跑模型訓練與效能評估

folderNames = [
    folder for folder, _ in pathToAPKs.items()
]

params = [
        "-i",
        str(list(pathToAPKs.items())[0][1]["path"]),
        "--force",
        "True",
        "--dbs",
        folderNames[0],
        "--dbs",
        folderNames[1],
        "--testonly",
        "False",
        "--device",
        "cuda",
        "--epoch",
        "500",
    ]

print(f"{params=}")

parser = parse_args(
    params
)

start(logger, parser)