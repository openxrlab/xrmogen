CONFIG=$1
PARTITION=$2
NUM_GPU=$3

SRUN="srun -p $PARTITION -n1 --mpi=pmi2 --gres=gpu:$NUM_GPU --ntasks-per-node=1 --cpus-per-task=8 --kill-on-bad-exit=0"
PYTHON="python -u "


echo "[INFO] Partition: $PARTITION, Used GPU Num: $NUM_GPU. "
echo "[INFO] SRUN: $SRUN"
echo "[INFO] PYTHON: $PYTHON"

SCRIPT1="main.py"

PYTHON_SCRIPT1="$PYTHON $SCRIPT1 --config $CONFIG --test_only"

echo "$SRUN $PYTHON_SCRIPT1"
$SRUN $PYTHON_SCRIPT1