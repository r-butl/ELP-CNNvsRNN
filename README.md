# Environment Setup 

If you are using the SDSC Expanse/ACCESS server, they provide a variety of containers for Tensorflow training. In my case, I needed a container with the Ray Tune package, which was unavailable. A nice work around to is to convert a prexisting container into a Singularity sandbox, run the sandbox container in a shell, add the needed packages, then launch scripts using that sandbox container.

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
singularity exec --writable sandbox_container/ bash
```

Now, execute pip to install Ray Tune, which is used for hyperparameter tuning of the model.
```
pip install -U "ray[data,train,tune,serve]" argparse optuna
```

Once this installation is complete, exit the shell and the contents should be saved:
```
exit
```

Now, we have a container we can use for training.

# Data Setup

First, locate the 'elp_audio_tfrecords' folder. This contains the training, testing, and validation audio samples of both elephant rumbles and non-elephant signals. If you are using the SDSC access server, the folder is unzipped and placed in /tmp/elp_audio_tfrecords (unless someone moved it).

If the file is zipped than unzip it. 
```
unzip tfrecords_cherrypicked.zip 
```

Now, its time to convert the audio samples into spectrogram samples. Move into the folder 'ELP-CNN-Spectrogram/data_creation' and run the following command to create the spectrogram files from the 'elp_audio_tfrecords' folder.

```
python convert_audio_samples_to_spectrogram.py --audio_tfrecords_directory <path-to>/elp_audio_tfrecords --output_folder ../spectrogram_records
```

Once the new files are created, make sure that the config.py file is configured to load the data correctly. This means making sure the path points to the correct folder and the file names match. Move into the ELP-CNN-Spectrogram directory, go into config.py and make sure that these parameters are set properly:

```
DATASET_FOLDER = './spectrogram_records'

TRAIN_FILE = 'spectrogram_train_cherrypick.tfrecord'
VALIDATE_FILE = 'spectrogram_validate_cherrypick.tfrecord'
TEST_FILE = 'spectrogram_test_cherrypick.tfrecord'
```

Once you have verified this, then we can run either the exploration, training or testing script.

# Exploration Script

The exploration script 'cross_validation_experiment.py' will try a variety of hyperparameters with the model using 5 fold cross validation. Ray Tune is used to scale up the experiments, running many at a time. The job script is the file 'run-cross_validation_experiment-gpu-shared.sh'. In order to queue this job on the gpu-shared partition of the server, run the following command:

```
sbatch run-cross_validation_experiment-gpu-shared.sh'
```

You can check the status of your job by running:

```
squeue -u $USER
```

This is an example output of running the previous commmand. The ST column refers to state, in this case the job is pending.

```             
JOBID       PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
37109179    gpu-debug cross_va lbutler2 PD       0:00      1 (Priority)
```
