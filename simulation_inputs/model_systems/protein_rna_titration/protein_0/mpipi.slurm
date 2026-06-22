#!/bin/bash
#SBATCH --job-name=SG_PR_System_0	  	# create a short name for your job
#SBATCH --nodes=1                		    # node count
#SBATCH --ntasks-per-node=96               	# total number of tasks across all nodes
#SBATCH --cpus-per-task=1        		    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         		    # memory per cpu-core (4G is default)
#SBATCH --time=24:00:00          		    # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all				        # send email on job start, end, and fail
#SBATCH --mail-user=your.email@example.edu	# optional: set your email address

module purge
 
module load intel/2021.1.2 
module load intel-mpi/intel/2021.3.1 

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

set -e 

rm output* || true
rm slurm-* || true
rm log.lammps*  || true
rm restart-* || true
rm result.lammpstrj* || true
rm print.statements* || true

srun --ntasks=96 ${LAMMPS_BIN:-lmp} -in lammps_mpipi_script.in > output0 2>&1

