#!/bin/bash
#SBATCH -n 16
#SBATCH -c 2
#SBATCH -t 001:00:00
#SBATCH -J Si-optical
#SBATCH -A xxxxx

module load QuantumESPRESSO/7.2-nsc1-intel-2018b-eb
export OMP_NUM_THREADS=2
thermo_pw.x <input.in | tee input.out
