#!/bin/bash
for SEED in `seq 1 5`
do
  python run_mujoco.py --logdir log_a2c_$SEED --env InvertedPendulum-v1 --relax False --numt 500000 --score 10000 --var_check True --seed $SEED &
  python run_mujoco.py --logdir log_a2c_relax_$SEED --env InvertedPendulum-v1 --relax True --numt 500000 --score 10000 --var_check True --seed $SEED &
done
