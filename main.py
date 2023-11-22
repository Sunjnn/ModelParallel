import os
import argparse

from torch import distributed, nn, optim
from torch.utils.data import DataLoader

from model import getResnet152 as getModel
from sampler import DistributedSampler
from datasets import getDataset
from parallelModel import ParallelModel


def addArguments():
    parser = argparse.ArgumentParser(description="main")
    parser.add_argument("--global_rank", type=int, default=0, help="global rank of all process")
    parser.add_argument("--global_size", type=int, default=1, help="global number of all process")
    parser.add_argument("--data_dir", type=str, default="./data", help="data directory which contains train and val sub directory")
    parser.add_argument("--out_dir", type=str, default="./out", help="output directory")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1", help="IP address of main process")
    parser.add_argument("--master_port", type=str, default="8888", help="IP port of main process")
    parser.add_argument("--GPU_ids", type=str, default="0", help="GPU ids that this process use")
    parser.add_argument("--num_layers", type=str, help="number of layers to each block")
    parser.add_argument("--batch_size", type=int, default=64, help="micro batch size")
    parser.add_argument("--mini_batch_size", type=int, help="number of mini batches that a pipeline execute")

    args = parser.parse_args()
    args.GPU_ids = list(args.GPU_ids.split(','))
    args.GPU_ids = [int(x) for x in args.GPU_ids]
    args.num_layers = list(args.num_layers.split(','))
    args.num_layers = [int(x) for x in args.num_layers]

    return args


def main():
    args = addArguments()

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    distributed.init_process_group(backend="nccl", world_size=args.global_size, rank=args.global_rank)

    model = getModel()
    train_sets = getDataset()
    sampler = DistributedSampler(train_sets, args.GPU_ids[0])
    train_loader = DataLoader(train_sets, batch_size=args.batch_size, sampler=sampler)
    parallel_model = ParallelModel(model, args.num_layers, nn.CrossEntropyLoss(), optim.SGD, args.GPU_ids, None)

    epoch = 100
    for i in range(epoch):
        parallel_model.train(train_loader, args.mini_batch_size)

    return 0


if __name__ == "__main__":
    main()
