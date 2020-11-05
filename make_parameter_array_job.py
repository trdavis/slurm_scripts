"""Constructs the script for a Slurm array job that performs a parameter sweep.

By default each parameter setting is assigned to its own parallel job. This can
be changed with the `runs_per_job` flag to reduce the total number of jobs, in
which case multiple parameter settings will be run sequentially in each job.

Requirements: python3.6+, numpy, absl-py

  Example usage:

  python make_parameter_array_job.py --out=script_name.sh \
      --param_linspace="foo,0,3,4" --param_linspace="bar,4.5,5.5,11" \
      -- my_script --arg1

      Creates script_name.sh that runs "my_script --arg1 --foo=FOO --bar=BAR"
      for all foo in (0, 1, 2, 3) and all bar in (4.5, 4.6, ..., 5.5)

  python make_parameter_array_job.py --out=script_name.sh \
      --param_logspace="foo,0.01,10,4" -- my_script --arg1

      Creates script_name.sh that runs "my_script --arg1 --foo=FOO"
      for all foo in (0.01, 0.1, 1, 10)
"""
import math
import pathlib
import shlex
import typing
from typing import Iterable, List

from absl import app
from absl import flags
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string("out", None, "Name for created job script file")
flags.mark_flag_as_required("out")
flags.DEFINE_string("dir",
                    str(pathlib.Path.home()) + "/scratch/jobscripts/",
                    "Base directory for job scripts")
flags.DEFINE_string("outdir",
                    str(pathlib.Path.home()) + "/scratch/output/",
                    "Base directory for job output")

HEADER_FLAGS = ["job-name", "output", "error", "time", "mem", "cpus-per-task"]
for header_flag in HEADER_FLAGS:
  flags.DEFINE_string(header_flag, None, f"Sets the Slurm flag '{header_flag}'")

flags.DEFINE_multi_string(
    "param_linspace", [], "Sweeps linearly spaced values for a parameter."
    ' Given "NAME,START,STOP,NUM" uses --NAME=x for NUM different x between '
    "START and STOP inclusive")
flags.DEFINE_multi_string(
    "param_logspace", [], "Sweeps geometrically spaced values for a parameter."
    ' Given "NAME,START,STOP,NUM" uses --NAME=x for NUM different x between '
    "START and STOP inclusive")
flags.DEFINE_multi_string(
    "param_list", [], "Explicit list of values for a parameter. Given in the"
    ' form "NAME,VAL1,VAL2[,VAL3,...]".')
flags.DEFINE_integer("runs_per_job", 1,
                     "Number of parameter to sequentially run in each job")


def validate_multiparam_str(param_str: str):
  """Validates parameter settings for linspace and logspace parameters.

  Checks that the value is of the form "NAME,START,STOP,NUM" where
  START and STOP are floats with START <= STOP and NUM is an integer.

  Args:
      param_str: The `str` value of the parameter to test.
  Returns:
      True if the parameter setting is valid.
  Raises:
      flags.ValidationError: The parameter setting is invalid.
  """
  vals = param_str.split(",")
  if len(vals) != 4:
    raise flags.ValidationError(
        'Array param value must be "NAME,START,STOP,NUM"')
  try:
    start = float(vals[1])
    stop = float(vals[2])
    num = int(vals[3])
  except ValueError as e:
    raise flags.ValidationError(
        'Array param "NAME,START,STOP,NUM" must have float'
        " START and STOP and int NUM") from e
  if start > stop:
    raise flags.ValidationError(
        'Array param "NAME,START,STOP,NUM" must have START <= STOP')
  if num < 1:
    raise flags.ValidationError(
        'Array param "NAME,START,STOP,NUM" must have NUM > 0')
  return True


def validate_all_multiparam(param_list: List[str]) -> bool:
  """Validates a list of parameter settings for linspace/logspace parameters.

  See validate_multiparam_str()

  Args:
      param_list: A list of str parameter values.
  Returns:
      True if the each parameter value is valid.
  Raises:
      flags.ValidationError: One or more parameter value is invalid.
  """
  for param_str in param_list:
    valid = validate_multiparam_str(param_str)
    if not valid:
      return valid
  return True


def validate_param_list_str(param_str: str):
  """Validates parameter settings for explicit parameter lists.

  Checks that the value is of the form "NAME,VAL1,VAL2[,VAL3,...]".

  Args:
      param_str: The `str` value of the parameter to test.
  Returns:
      True if the parameter setting is valid.
  Raises:
      flags.ValidationError: The parameter setting is invalid.
  """
  vals = param_str.split(",")
  if len(vals) < 3:
    raise flags.ValidationError(
        'param_list value must be "NAME,VAL1,VAL2[,VAL3,...]"')
  return True


def validate_all_param_list(param_list: List[str]) -> bool:
  """Validates a list of explicitly listed parameters.

  See validate_param_list_str()

  Args:
      param_list: A list of str parameter values.
  Returns:
      True if the each parameter value is valid.
  Raises:
      flags.ValidationError: One or more parameter value is invalid.
  """
  for param_str in param_list:
    valid = validate_param_list_str(param_str)
    if not valid:
      return valid
  return True


flags.register_validator("param_linspace", validate_all_multiparam)
flags.register_validator("param_logspace", validate_all_multiparam)
flags.register_validator("param_list", validate_all_param_list)


class ParamArray(typing.NamedTuple):
  """Collects a parameter name with a list of possible values.

  Attributes:
      param: The parameter name.
      values: A list of str parameter values.
  """
  param: str
  values: List[str]


def parse_param_array(param_str: str, linspace: bool = True) -> ParamArray:
  """Converts a linspace/logspace parameter string into an array of values.

  The str should be in format "NAME,START,STOP,NUM".
  See validate_multiparam_str() for more details.

  Args:
      param_str: The str value of the parameter.
      linspace: If true, specifies a linearly spaced array. If false, specifies
          a geometrically spaced array.
  Returns:
      A `ParamArray` containing the list of parameter values.
  """
  flag_vals = param_str.split(",")
  assert len(flag_vals) == 4
  start = float(flag_vals[1])
  stop = float(flag_vals[2])
  num = int(flag_vals[3])

  if linspace:
    param_vals = np.linspace(start, stop, num)
  else:
    param_vals = np.logspace(math.log10(start), math.log10(stop), num)

  return ParamArray(flag_vals[0], [str(x) for x in param_vals])


def parse_all_param_arrays(param_list: List[str],
                           linspace: bool = True) -> List[ParamArray]:
  """Parses a list of parameter str values into arrays.

  See parse_parameter_array()

  Args:
      param_list: A list of str values for different parameters.
      linspace: If true, the strings are for linearly spaced arrays.
          If false, geometrically spaced arrays.
  Returns:
      A list of `ParamArray` containing all parameter values.
  """
  return [parse_param_array(s, linspace) for s in param_list]


def parse_explicit_param_list(param_str: str) -> ParamArray:
  """Parses parameter values explicitly listed in str form.

  The str should be in form "NAME,VAL1,VAL2[,VAL3,...]".

  Args:
      param_str: The str value of the parameter, in form
          "NAME,VAL1,VAL2[,VAL3,...]"
  Returns:
      A `ParamArray` containing the parameter values.
  """
  tokens = param_str.split(",")
  assert len(tokens) > 2
  return ParamArray(tokens[0], tokens[1:])


def parse_all_param_lists(param_lists: List[str]) -> List[ParamArray]:
  """Parses a list of param_list values into arrays.

  See parse_explicit_param_list()

  Args:
      param_lists: A list of param_list strs for different parameters.
  Returns:
      A list of `ParamArray` containing all parameter values.
  """
  return [parse_explicit_param_list(s) for s in param_lists]


def script_header(num_jobs: int) -> List[str]:
  """Constructs the header of the array job script.

  Args:
      num_jobs: The number of parallel jobs in the job array.
  Returns:
      A list of header lines for the script.
  """
  assert num_jobs >= 1
  header = []
  header.append("#!/bin/bash")
  if num_jobs > 1:
    header.append(f"#SBATCH --array=1-{num_jobs}")
  for flag in HEADER_FLAGS:
    if FLAGS[flag].value:
      header.append(f"#SBATCH --{flag}={FLAGS[flag].value}")
  return header


def script_param_arrays(param_arrays: Iterable[ParamArray]) -> List[str]:
  """Constructs the parameter arrays used in the array job script.

  Args:
      param_arrays: A list of `ParamArray` representing all possible values
          for the gridsearch parameters.
  Returns:
      A list of array initialization lines for the script.
  """
  result = []
  for param_array in param_arrays:
    result.append(f"{param_array.param}=({' '.join(param_array.values)})")
  return result


def script_param_indexing(array_lengths: Iterable[int]) -> List[str]:
  """Creates the script lines that set parameter indices for a run.

  Args:
      array_lengths: The number of parameter settings used for each parameter,
          in order.
  Returns:
      A list of script lines that set the parameter indices.
  """
  indexing = []
  indexing.append("idx=${run_index}")
  for i, l in reversed(list(enumerate(array_lengths))):
    indexing.append(f"param_index[{i}]=$((idx % {l}))")
    indexing.append(f"((idx /= {l}))")
  return indexing


def script_run(command: str, param_names: Iterable[str],
               array_job: bool) -> List[str]:
  """Creates the script lines that run a single parameter setting.

  Args:
      command: The base program string to be executed, without parameter
          settings.
      param_names: A list of names for parameters that vary between runs.
      array_job: True if the runs are split between more than one job.
  Returns:
      A list of script runs that set up and execute the run.
  """
  output = []
  params = []
  for i, name in enumerate(param_names):
    params.append(f"--{name}=${{{name}[${{param_index[{i}]}}]}}")
  param_str = " ".join(params)
  run = [command, param_str]
  if FLAGS.outdir:
    if array_job:
      outfile = (f"{FLAGS.outdir}${{SLURM_JOB_NAME}}"
                 "-${SLURM_ARRAY_JOB_ID}-r${run_index}.out")
    else:
      outfile = (f"{FLAGS.outdir}${{SLURM_JOB_NAME}}"
                 "-${SLURM_JOB_ID}-r${run_index}.out")
    output.append(f'outfile="{outfile}"')
    output.append('echo "${command}" >> ${outfile}')
    output.append(f'echo "# {param_str}" | tee -a ${{outfile}}')
    run.append(" | tee -a ${outfile}")
  else:
    output.append(f'echo "# {param_str}"')
  run_str = " ".join(run)
  output.append(run_str)
  return output


def script_loop(command: str, param_arrays: List[ParamArray], runs_per_job: int,
                total_runs: int) -> List[str]:
  """Constructs the script loop that executes all runs for a job.

  Args:
      command: The base program string to be executed, without parameter
          settings.
      param_arrays: A list of `ParamArray` that gives the settings for
          parameters that are varied across runs. Contains the settings
          for all jobs, not just the current one.
      runs_per_job: The number of runs executed in each individual job.
      total_runs: The number of runs executed across all jobs.
  Returns:
      A list of script lines for executing all job runs.
  """
  loop = []
  initial_index_array = ["0"] * len(param_arrays)
  loop.append(f"param_index=({' '.join(initial_index_array)})")
  loop.append(f'command="# {command}"')
  loop.append("echo ${command}")
  loop.append(f"for ((run=0; run < {runs_per_job}; run++)); do")
  inner = []
  if runs_per_job < total_runs:
    inner.append(
        f"run_index=$(({runs_per_job}*$((${{SLURM_ARRAY_TASK_ID}} - 1))"
        " + ${run}))")
  else:
    inner.append("run_index=${run}")
  inner.append(f'if [[ "${{run_index}}" -ge {total_runs} ]]; then')
  inner.append("  break")
  inner.append("fi")
  inner += script_param_indexing(len(pa.values) for pa in param_arrays)
  inner.append("")
  inner += script_run(command, (pa.param for pa in param_arrays),
                      (runs_per_job < total_runs))
  loop += ["  " + line for line in inner]
  loop.append("done")
  return loop


def main(argv):
  param_arrays = (parse_all_param_arrays(FLAGS.param_linspace, linspace=True) +
                  parse_all_param_arrays(FLAGS.param_logspace, linspace=False) +
                  parse_all_param_lists(FLAGS.param_list))
  total_runs = math.prod(len(pa.values) for pa in param_arrays)
  runs_per_job = FLAGS.runs_per_job
  num_jobs = (total_runs + (runs_per_job - 1)) // runs_per_job  # ceiling div
  if FLAGS.outdir:
    outdir = pathlib.Path(FLAGS.outdir)
    if not outdir.is_dir():
      print("Invalid output directory", outdir)
      return -1
    outdir = outdir.resolve()
    if not FLAGS.output:
      if num_jobs > 1:
        FLAGS.output = str(outdir / "%x-%A-j%a.out")
      else:
        FLAGS.output = str(outdir / "%x-%j-j0.out")
    if not FLAGS.error:
      if num_jobs > 1:
        FLAGS.error = str(outdir / "%x-%A-j%a.err")
      else:
        FLAGS.error = str(outdir / "%x-%j-j0.err")
  script = script_header(num_jobs)
  script.append("")
  script += script_param_arrays(param_arrays)
  script.append("")
  script += script_loop(" ".join(shlex.quote(s) for s in argv[1:]),
                        param_arrays, runs_per_job, total_runs)
  scriptdir = pathlib.Path(FLAGS.dir)
  if not scriptdir.is_dir():
    print("Invalid script directory", scriptdir)
    return -1
  script_path = scriptdir / FLAGS.out
  with script_path.open("w") as jobscript:
    print("\n".join(script), file=jobscript)


if __name__ == "__main__":
  app.run(main)
