#!/bin/bash
#SBATCH -A e30514
#SBATCH -p short
#SBATCH --job-name="rpstest"
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -t 00:00:10
#SBATCH --mem=8G

cd $SLURM_SUBMIT_DIR
module load mpi/openmpi-4.1.7-gcc-10.4.0
module load blas-lapack/3.12.0-gcc-11.2.0

mpirun -n 2 ./rps 128 9.0 1 1