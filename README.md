# SDSC Setup

If you are using the SDSC Expanse/ACCESS server, they provide a variety of containers for Tensorflow training. In my case, I needed a container with the Ray Tune package, which was unavailable. If you are too lazy to get permission to build an image, a nice work around to is to convert a prexisting container into a Singularity sandbox, run the sandbox container in a shell, add the needed packages, then launch scripts using that sandbox container.

First, load the singularity module:
```
module load singularitypro
```

Then, build the container into a Singularity sandbox. The output of this command will be a directory named 'sandbox_container':
```
singularity build --sandbox sandbox_container/ /cm/shared/apps/containers/singularity/tensorflow/tensorflow-latest.sif
```

Once the container has been built into a sandbox directory, load the container in a shell:
```
singularity exec --writable train-container-sandbox/ /bin/bash
```

Now, execute pip to install the necessary packages.
```
pip install -r requirements.txt
```

Once this installation is complete, exit the shell and the contents should be saved:
```
exit
```

Now, we have a container we can use for training.

To log in directly to the container, use this command:
```
singularity exec --bind /expanse,/scratch --nv ./train-container-sandbox /bin/bash
```

# Loading Singularity Modules

View what modules are available to load:
```
module avail
```

Terminal output:
```
------------------------------------ /cm/shared/apps/spack/0.17.3/cpu/b/share/spack/lmod/linux-rocky8-x86_64/Core ------------------------------------
   anaconda3/2021.05/q4munrg             gcc/10.2.0/npcyll4          matlab/2022b/lefe4oq         rclone/1.56.2/mldjorr
   aocc/3.2.0/io3s466                    gh/2.0.0/mkz3uxl            mercurial/5.8/qmgrjvl        sratoolkit/2.10.9/rn4humf
   aria2/1.35.0/q32jtg2                  git-lfs/2.11.0/kmruniy      nvhpc/21.9/xxpthf5           subversion/1.14.0/qpzq6zs
   cmake/3.21.4/n5jtjsf                  git/2.31.1/ldetm5y          parallel/20210922/sqru6rr    ucx/1.10.1/wla3unl
   entrezdirect/10.7.20190114/6pkkpx2    intel/19.1.3.304/6pv46so    pigz/2.6/bgymyil
--------------------------------------------------------------- /cm/local/modulefiles ----------------------------------------------------------------
   shared (L)    singularitypro/3.11 (D)    singularitypro/4.1.2    slurm/expanse/23.02.7 (L)
--------------------------------------------------------- /cm/shared/apps/access/modulefiles ---------------------------------------------------------
   accessusage/0.5-1    cue-login-env
--------------------------------------------------------------- /usr/share/modulefiles ---------------------------------------------------------------
   DefaultModules (L)    cpu/0.15.4 (c)    cpu/0.17.3b (c,L,D)    gpu/0.15.4 (g)    gpu/0.17.3b (g,D)    nostack/0.15.4 (e)    nostack/0.17.3b (e,D)
--------------------------------------------------------------- /cm/shared/modulefiles ---------------------------------------------------------------
   AMDuProf/3.4.475    default-environment    sdsc/1.0 (L)    slurm/expanse/23.02.7
  Where:
   L:  Module is loaded
   c:  built natively for AMD Rome
   e:  not architecture specific
   g:  built natively for Intel Skylake
   D:  Default Module
Module defaults are chosen based on Find First Rules due to Name/Version/Version modules found in the module tree.
See https://lmod.readthedocs.io/en/latest/060_locating.html for details.
Use "module spider" to find all possible modules and extensions.
Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".
```


Adds a module to your environment for this session:
```
module add <module>
``` 

For singularity, you will want to load this one:
```
module add singularitypro/3.11
```

You can configure module to be loaded at every login:
```
echo "module load singularitypro/3.11" >> ~/.bashrc
```

Viewing loaded modules:
```
module list
```

# Launch a terminal with allocation

Here is an example command you can use to launch a bash terminal with GPU allocation. 

```
srun --partition=gpu-shared --gpus=1 --pty --account=cso100 --nodes=1 --ntasks-per-node=4 --mem=8G -t 04:00:00 --export=ALL singularity exec --bind /expanse,/scratch,/usr/share/lmod:/usr/share/lmod,/usr/bin/lua:/usr/bin/lua --nv ./train-container-sandbox /bin/bash
```

```
sbatch --partition=gpu-shared --gpus=1 --account=cso100 --nodes=1 --ntasks-per-node=4 --mem=32G -t 06:00:00 --export=ALL singularity exec --bind /expanse,/scratch,/usr/share/lmod:/usr/share/lmod,/usr/bin/lua:/usr/bin/lua --nv ./train-container-sandbox python scripts/cross_validation.py --model mobilenetv2 --config config.yaml
```

Train:
```
python scripts/train_regular.py --model resnet18 --config config.yaml --best_config /home/lbutler2/ELP-CNN-Spectrogram/cross_validation_results/resnet18_cv_results/resnet18_best_config/best_config.json  
```

# Queueing a Job
To queue a job to run on the gpu-shared cluster, you will need to define a bash script with the following outline:

```
#!/usr/bin/env bash
#SBATCH --job-name=cross_validation_experiment
####Change account below
#SBATCH --account=cso100
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=%x.o%j.%N
declare -xr SINGULARITY_MODULE='singularitypro/3.11'
module purge
module load "${SINGULARITY_MODULE}"
module list
# Check if model type argument is passed
if [ -z "$1" ]; then
    echo "Error: No model type specified. Usage: sbatch $0 <cnn|rnn>"
    exit 1
fi
MODEL_TYPE=$1  # Capture model type argument
export NVIDIA_DISABLE_REQUIRE=true
time -p singularity exec --bind /expanse,/scratch --nv ./train-container-sandbox python -u ./cross_validation_experiment.py --model "$MODEL_TYPE"
```

Then you can use this command to schedule it.
```
sbatch scripts/run-cross_validation_experiment-gpu-shared.sh
```

*** NOTE: This script is currently not a part of the project, this is just an example. ***

# Testing the Pipeline

After you have loaded the singularity module, made a conda environment, and launched a bash terminal with GPU allocation, you can then procede to test the scripts. The first action is to make the test dataset.

```
python test_pipeline.py
```

# Queueing and Viewing Jobs

You can check the status of your job by running:

```
squeue -u $USER
```

or 
```
squeue | grep $USER
```

This is an example output of running the previous commmand. The ST column refers to state, in this case the job is pending. When in the running state, the target nodes can be found unnder the Nodelist column.

```
JOBID    PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
42897218 gpu-debug     bash lbutler2  R       4:21      1 exp-7-59
```


Check if the script is properly using the GPU, ssh into a node in the Nodelist column.
```
ssh <node>
nvtop
```