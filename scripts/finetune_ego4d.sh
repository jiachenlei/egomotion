# total number of gpus per node
GPUS_PER_NODE=2
# total number of nodes
NODE=1
# global rank of current process
GLOBAL_RANK=$1
# address of master process whose global rank equals 0
MASTER_ADDR=$2
# port of master process whose global rank equals 0, default value is 12345
MASTER_PORT=12345

# name of the finetuning experiment. default value is "temp"
NAME="temp"

OMP_NUM_THREADS=40 python -m torch.distributed.launch \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port $MASTER_PORT --nnodes=$NODE  --node_rank=$GLOBAL_RANK --master_addr=$MASTER_ADDR \
    ../run_class_finetuning.py \
    --enable_deepspeed \
    --dist_eval \
    --overwrite command-line \
    --config /mnt/code/egomotion/config/finetune_vitb_ego4d.yml \
    --project finetune_ego4d \
    --name $NAME \
    --debug # comment this line, then log information will be uploaded to wandb
