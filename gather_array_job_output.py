#!/bin/python3
"""Combines output values from each job in an array into a single file.

The array job is assumed to be created with make_parameter_array_job.py
This script then reads in all output files from individual parameter runs
in the array. The final numerical value in the output is assumed to be a
performance record. These values are output in a whitespace-delimited file with
columns giving the parameter settings for each run.

This script avoid using any imports other than the standard library so that it
can run in Compute Canada's default environment. If other modules are required,
this script will need to be wrapped in a shell script that creates/loads a
virtual environment.

    Example usage:

    gather_array_job_output.py 1234567 ~/scratch/output/ ~/data/
"""
import pathlib
import operator
import re
import sys
import typing
from typing import Callable, Dict, IO, List, Optional, Tuple, Union

# Column name given to values extracted from output
VALUE_NAME = "Exploitability"

DataValue = Union[str, int, float]


class OutputFileRawData(typing.NamedTuple):
  """Collects data from an array job output file.

  Attributes:
      command: The base command used to produce the output file
      params: Additional parameter settings particular to the specific run
              which produced the output file
      data_lines: A list of str data lines output to the file
  """
  command: str
  params: Dict[str, DataValue]
  data_lines: List[str]


class OutputData(typing.NamedTuple):
  """Collects the aggregated data from all output files.

  Attributes:
      command: The base command used to produce the output files
      parameter_names: The names of parameters specifically set in individual
                       output files
      value_names: The names of values parsed from any output file
      records: A list of individual data points, each of which maps parameters
               and values to settings. Each output file can produce one or more
               data points.
  """
  command: str
  parameter_names: List[str]
  value_names: List[str]
  records: List[Dict[str, DataValue]]


def print_usage(name):
  """Prints script usage message to stderr.

  Args:
      name: invoking name for script
  """
  print("Usage:", name, "ARRAY_JOB_ID OUTDIR DATADIR", file=sys.stderr)


float_regex = re.compile(
    r"[-+]? (?:(?: \d* \. \d+ ) | (?: \d+ \.? ))"
    r"(?: [Ee] [+-]? \d+ )?", re.VERBOSE)
param_regex = re.compile(r"--(\w[^\s=]*)=(\S+)")


def output_files(output_dir: pathlib.Path,
                 jobid: str) -> Tuple[List[pathlib.Path], Optional[str]]:
  """Gives the list of output files from an array job.

  Args:
      output_dir: The directory containing output files
      jobid: A str containing an integer slurm job ID
  Returns:
      A tuple containing:
      - The list of all output files matching the specified jobid
      - The name of the job that produced the output, or None if a name could
        not be parsed
  """
  files = list(output_dir.glob("*-{}-r*.out".format(jobid)))
  if not files:
    return [], None
  name_match = re.search("(.+)-" + jobid, files[0].name)
  if not name_match:
    return files, None
  job_name = name_match.groups()[0]
  return files, job_name


def file_data(file_path: pathlib.Path) -> Optional[OutputFileRawData]:
  """Reads data from a file representing a single run.

  The first line should start with # and give the base command.
  Parameter settings are read from header lines (ignoring the initial line) of
  the form
      # --param1=value --param2=value

  Args:
      file_path: A pathlib.Path pointing to the file to process
  Returns:
      A OutputFileRawData object containing the file's data, or None if a
      parsing error occurs
  """
  with file_path.open() as outfile:
    lines = outfile.readlines()
  lines = [line.strip() for line in lines]

  command = None
  params = {}
  for line in lines:
    if not line:
      continue
    if line[0] != "#":
      break
    if not command:
      command = line[1:].strip()
    else:
      matches = param_regex.findall(line)
      if not matches:
        continue
      for param, setting in matches:
        if not param or not setting:
          print("Failed to parse parameter name and value from line '{}'".format(
              line))
          return None
        # Try to convert param setting to int/float. This allows proper sorting
        try:
          setting = int(setting)
        except ValueError:
          try:
            setting = float(setting)
          except ValueError:
            pass
        params[param] = setting

  return OutputFileRawData(command, params,
                           [line for line in lines if line[0] != "#"])


def output_data(
    files : List[pathlib.Path],
    parser : Callable[[List[str]], List[Dict[str, DataValue]]]
) -> Optional[OutputData]:
  """Parses output datapoints from a list of output files.

  Args:
      files: A list of data output files to parse
      parser: A function that turns a list of data lines into a list of data
              points. Each data point is a dict that maps parameter and value
              names to their values. For an example parser, see last_value().
  Returns:
      An OutputData object summarizing the data, or None if an error occurs.
  """
  records = []
  parameter_names = []
  value_names = []
  command = None
  for file_path in files:
    assert file_path.is_file()
    data = file_data(file_path)
    if not data:
      return None
    if command is not None and data.command != command:
      print("Warning: command mismatch between output files", file=sys.stderr)
    command = data.command
    values_list = parser(data.data_lines)
    if values_list is None:
      return None
    if not values_list:
      print("Warning: no values found for output", file_path, file=sys.stderr)
    for param in data.params.keys():
      if param not in parameter_names:
        parameter_names.append(param)
    for values in values_list:
      for value_name in values.keys():
        if value_name not in value_names:
          value_names.append(value_name)
      if data.params.keys() & values.keys():
        print("Warning: overlap between parameter and value names in",
              file_path, file=sys.stderr)
      records.append({**data.params, **values})
  return OutputData(command, parameter_names, value_names, records)


def write_data(data : OutputData, outfile : IO[str]) -> None:
  """Writes the aggregated data to a single output file.

  Each data point is written as a whitespace-delimited line. The first line
  specifies the command after comment character #. The second line gives the
  column names, which are the parameter/value names.

  Args:
      data: OutputData aggregated from all of the original output files
      outfile: An I/O text stream to output to
  """
  columns = data.parameter_names
  columns += [x for x in data.value_names if x not in columns]
  records = sorted(data.records, key=operator.itemgetter(*columns))
  max_width = max(len(c) for c in columns)
  max_width = max(max_width, 7)
  str_format = "{:" + str(max_width) + "." + str(max_width) + "}"
  float_format = "{:" + str(max_width) + "." + str(max_width - 3) + "}"
  int_format = "{:" + str(max_width) + "}"

  def format_str(s):
    return str_format.format(s)

  def format_float(f):
    return float_format.format(f)

  def format_int(i):
    if len(str(i)) <= max_width:
      return int_format.format(i)
    else:
      return format_float(float(i))

  print("#", data.command, file=outfile)
  print(" ".join(format_str(c) for c in columns), file=outfile)
  for record in records:
    entries = []
    for column in columns:
      entry = record.get(column, "N/A")
      if isinstance(entry, int):
        formatted = format_int(entry)
      elif isinstance(entry, float):
        formatted = format_float(entry)
      else:
        formatted = format_str(entry)
      entries.append(formatted)
    print(" ".join(entries), file=outfile)


def last_value(
    data_lines : List[str],
    value_name : str = VALUE_NAME) -> List[Dict[str, DataValue]]:
  """Gives the last float value in a set of output lines.

  Args:
      data_lines: Lines of data output
      value_name: The name for what the value represents, i.e. the y-xaxis label
  Returns:
      An empty list if no value is found. Otherwise, a list containing a single
      dict mapping the value name to the last valid float that appears in the
      output, i.e. the last value in the last line
  """
  value = None
  for line in reversed(data_lines):
    line = line.strip()
    if not line or line[0] == "#":
      continue
    matches = float_regex.findall(line)
    if matches:
      value = float(matches[-1])
      break
  if not value:
    return []
  return [{value_name : value}]


def main(argv):
  if len(argv) != 4:
    print_usage(argv[0])
    return -1
  try:
    jobid = argv[1]
    int(jobid)
  except ValueError:
    print_usage(argv[0])
    print("Couldn't parse integer jobid", file=sys.stderr)
    return -1
  outdir = pathlib.Path(argv[2])
  datadir = pathlib.Path(argv[3])
  if not outdir.is_dir() or not datadir.is_dir():
    print_usage(argv[0])
    return -1

  files, job_name = output_files(outdir, jobid)
  if not files:
    print("No files found ", file=sys.stderr)
    return -1
  if job_name is None:
    print("Failed to parse job name from output files", file=sys.stderr)
    return -1

  data = output_data(files, last_value)
  if not data:
    return -1

  data_path = datadir / job_name
  if data_path.exists():
    data_path = datadir / "{}-{}".format(job_name, jobid)
    print("Default data file exists. Writing to {} instead".format(data_path))

  with data_path.open("w") as datafile:
    write_data(data, datafile)


if __name__ == "__main__":
  main(sys.argv)
