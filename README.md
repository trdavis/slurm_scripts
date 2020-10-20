# slurm_scripts
Scripts for running jobs on Slurm

Currently contains three scripts I use for performing parameter sweeps/grid searches:
- `make_parameter_array_job.py` constructs a bash script for a Slurm array job that runs an executable for every combination of specified parameter settings.
- `process_array_job.py` submits a Slurm (array) job and a processing job that will not run until the original job completely finishes. The processing job is used to collect output from all of the runs with different parameter settings.
- `gather_array_job_output.py` is the script I use for the processing job. It reads a value from each output file and outputs a single data file summarizing the results.

For details, see the documentation in each script.

## Example Usage
```bash
python make_parameter_array_job.py --out=script_name.sh --param_linspace="foo,0,5,6" \
    --param_logspace="bar,0.01,10,4" -- parametric_executable --flag=value positional_argument
python process_array_job.py --jobscript=$HOME/scratch/jobscripts/script_name.sh --process=process_array_job.py
```
