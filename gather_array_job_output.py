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
from typing import Dict, List, Tuple, Union

# Column name given to values extracted from output
VALUE_NAME = "Exploitability"


def print_usage(name):
  """Prints script usage message to stderr.

  Args:
      name: invoking name for script
  """
  print("Usage:", name, "ARRAY_JOB_ID OUTDIR DATADIR", file=sys.stderr)


float_regex = re.compile(
    r"[-+]? (?:(?: \d* \. \d+ ) | (?: \d+ \.? ))"
    r"(?: [Ee] [+-]? \d+ )?", re.VERBOSE)
param_regex = re.compile(r"\# \s* -- (\w+) = (\S+)", re.VERBOSE)


def file_data(
    file_path: pathlib.Path, columns: List[str]
) -> Union[None, Tuple[Dict[str, Union[str, int, float]], str]]:
  """Reads data from a file representing a single run.

  Parameter settings are read from header lines (ignoring the initial line) of
  the form

      # --param=value

  Args:
      file_path: A pathlib.Path pointing to the file to process
      columns: A list of column names for the final output file. The name of any
          parameter set in the file's header will be added to the list.
  Returns:
      A tuple containing:
      - A dictionary mapping column names to values for this particular run.
        This includes all parameter settings as well as the value read from
        output.
      - A string giving the command used to produce the output, without
        parameter settings. This is parsed from the first line of output.
  """
  with file_path.open() as outfile:
    lines = outfile.readlines()

  # Find last float value in program output
  for line in reversed(lines):
    line = line.strip()
    if not line or line[0] == "#":
      continue
    matches = float_regex.findall(line)
    if matches:
      value = float(matches[-1])
      break
  if not value:
    print("Failed to parse value from file", file_path, file=sys.stderr)
    return None

  file_record = {VALUE_NAME: value}

  # Read param settings from header
  for line in lines[1:]:
    match = param_regex.search(line)
    if not match:
      break
    param, setting = match.groups()
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
    file_record[param] = setting
    if param not in columns:
      columns.insert(-1, str(param))
  return file_record, lines[0].strip()


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

  output_files = list(outdir.glob("*-{}-r*.out".format(jobid)))
  if not output_files:
    print("Failed to find matching output files", file=sys.stderr)
    return -1
  name_match = re.search("(.+)-" + jobid, output_files[0].name)
  if not name_match:
    print("Failed to parse job name from", output_files[0])
    return -1
  job_name = name_match.groups()[0]

  records = []
  columns = [VALUE_NAME]
  command = None
  for output_file in output_files:
    file_path = outdir / output_file
    assert file_path.is_file()
    data = file_data(file_path, columns)
    if not data:
      return -1
    file_record, file_command = data
    if command is not None and file_command != command:
      print("Warning: command mismatch between output files", file=sys.stderr)
    command = file_command
    if not file_record:
      return -1
    records.append(file_record)

  for record in records:
    for column in columns:
      if column not in record:
        print("Record missing setting for", column, file=sys.stderr)
        print(record, file=sys.stderr)
        return -1

  records.sort(key=operator.itemgetter(*columns))
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

  data_path = datadir / job_name
  if data_path.exists():
    data_path = datadir / "{}-{}".format(job_name, jobid)
    print("Default data file exists. Writing to {} instead".format(data_path))
  with data_path.open("w") as datafile:
    print(command, file=datafile)
    print(" ".join(format_str(c) for c in columns), file=datafile)
    for record in records:
      entries = []
      for column in columns:
        entry = record[column]
        if isinstance(entry, int):
          formatted = format_int(entry)
        elif isinstance(entry, float):
          formatted = format_float(entry)
        else:
          formatted = format_str(entry)
        entries.append(formatted)
      print(" ".join(entries), file=datafile)


if __name__ == "__main__":
  main(sys.argv)
