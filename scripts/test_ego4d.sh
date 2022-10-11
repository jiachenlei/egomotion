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

NAME="temp"  # name of the experiment. it should equal to the finetuning experiment name. default value is "temp"
CHECKPOINT="" # path to checkpoint to be tested

# path to files that contain prediction result
# e.g. if two gpus are used, then 
# PATH=( /path/to/output_dir/temp/0.txt /path/to/output_dir/temp/1.txt)
# where 0 and 1 represent global rank
RESULT_FILE_PATH=()

# path to fho_oscc-pnr_test_unannotated.json
ANNOTATION_FILE="/path/to/fho_oscc-pnr_test_unannotated.json"

OMP_NUM_THREADS=40 python -m torch.distributed.launch \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port $MASTER_PORT --nnodes=$NODE  --node_rank=$GLOBAL_RANK --master_addr=$MASTER_ADDR \
    ../test_on_ego4d.py \
    --overwrite command-line \
    --name $NAME \
    --config /mnt/code/egomotion/config/test_basic_ego4d.yml \
    --ckpt $CHECKPOINT \
    --dist_eval \


# merge prediction, and specified *.json files will be saved in same directory as file proviede in $PATH
# by default the number of spatial crop is 3
python ../merge_test_result.py \
    --path ${RESULT_FILE_PATH[@]} \
    --num_crop 3 \
    --annotation_file $ANNOTATION_FILE