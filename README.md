# ModelParallel

Execute parallel computing for train deep learning model on distributed heterogeneous GPU systems,
that is several servers each equipped with a number of homogeneous GPU and GPU type is different across servers.

In a server,
some GPUs comprise of a pipeline parallel system and others comprise of other.
This can be assigned in command line options.
A single process manage a whole pipeline parallel system as GPU operations are asynchronously.
Data parallel are used to synchronize parameters between difference pipeline parallel systems.

## run

To run the project,
firstly GPUs in a server should be divided into several pipeline parallel system (one system is ok),
and model should also be divided into several blocks.
Note that difference system should have same architecture both in GPU and model.

For each system,
a command should run as follow.

``` bash
# global_rank is rank of this process among all processes
# global_size is number of all processes
# data_dir is data directory that should contains train and val sub directory
# out_dir is directory of output
# master_addr and master_port are IP address and port that master process bind
# GPU_ids is GPU id that this process use, for example, 0,1,2,3
# num_layers is number of layers that each block contains
# batch_size is micro batch size
# mini_batch_size is mini batch size
python main.py --global_rank RANK \
               --global_size GLOBAL_SIZE \
               --data_dir DATA_DIR \
               --out_dir OUT_DIR \
               --master_addr MASTER_ADDR \
               --master_port MASTER_PORT \
               --GPU_ids GPU_IDS \
               --num_layers NUM_LAYERS \
               --batch_size BATCH_SIZE \
               --mini_batch_size MINI_BATCH_SIZE
```
