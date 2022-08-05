CONFIG=$1
PARTITION=$2
NUM_GPU=$3

SRUN="srun -p $PARTITION -n1 --mpi=pmi2 --gres=gpu:$NUM_GPU --ntasks-per-node=1 --cpus-per-task=8 --kill-on-bad-exit=0"
PYTHON="python -u "


echo "[INFO] Partition: $PARTITION, Used GPU Num: $NUM_GPU. "
echo "[INFO] SRUN: $SRUN"
echo "[INFO] PYTHON: $PYTHON"

SCRIPT1="run_mogen.py"

PYTHON_SCRIPT1="$PYTHON $SCRIPT1 --config $CONFIG 
# PYTHON_SCRIPT1="$PYTHON $SCRIPT1 --config ./configs/kilonerfsv3/$CONFIG --test_only"

echo "$PYTHON_SCRIPT1"
$PYTHON_SCRIPT1

# srun -p dsta -n1 --mpi=pmi2 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=8 python main.py --config ./configs/dance_rev.py 