#!/bin/python3
"""Combines all output values from each job in an array into a single file.

The array job is assumed to be created with make_parameter_array_job.py
This script then reads in all output files from individual parameter runs
in the array. All lines in the file are assumed to contain labelled values,
i.e. "foo 3.1415" or "bar=3.1415". These values are output in a
whitespace-delimited file with additional columns giving the parameter settings
for each run.

This script avoid using any imports other than the standard library so that it
can run in Compute Canada's default environment. If other modules are required,
this script will need to be wrapped in a shell script that creates/loads a
virtual environment.

    Example usage:

    gather_all_array_job_output.py 1234567 ~/output/ ~/data/
"""
import pathlib
import operator
import re
import sys
import typing
from typing import Callable, Dict, IO, List, Optional, Tuple, Union
import gather_array_job_output as gather


def parse_value(value_str : str) -> gather.DataValue:
  """Parses a str as float, int, or str.

  Args:
      value_str: the str to parse
  Returns:
      A parsed int if possible, or a parsed float if possible, or the str
  """
  if len(value_str) >= 2:
    # remove quotes from parsed str
    if value_str[0] in {"'", '"'} and value_str[0] == value_str[-1]:
      return value_str[1:-1]
  try:
    value = int(value_str)
  except ValueError:
    try:
      value = float(value_str)
    except ValueError:
      value = value_str
  return value


def all_values(data_lines : List[str]) -> List[Dict[str, gather.DataValue]]:
  """Gives all labelled output values in an output file.

  Args:
      data_lines: Lines of data output
  Returns:
      A list of data points. Each data point is a dict mapping parsed value
      names to parsed values.
  """
  data = []
  for line in data_lines:
    line = line.strip()
    if not line or line[0] == "#":
      continue
    record = {}
    words = line.split()
    i = 0
    while i < len(words):
      tokens = words[i].split("=")
      if len(tokens) == 1:
        value_name = words[i]
        if i + 1 == len(words):
          print("Couldn't find value token for", value_name, file=sys.stderr)
          break
        value_str = words[i + 1]
        i += 2
      elif len(tokens) == 2:
        value_name = tokens[0]
        value_str = tokens[1]
        i += 1
      else:
        print("Too many = in output", words[i], file=sys.stderr)
        i += 1
        continue
      value = parse_value(value_str)
      if value_name in record:
        print("Warning: {} appears multiple times in line".format(value_name),
              file=sys.stderr)
      record[value_name] = value
    if record:
      data.append(record)
  return data

def print_usage(name):
  """Prints script usage message to stderr.

  Args:
      name: invoking name for script
  """
  print("Usage:", name, "ARRAY_JOB_ID OUTDIR DATADIR", file=sys.stderr)

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

  files, job_name = gather.output_files(outdir, jobid)
  if not files:
    print("No files found ", file=sys.stderr)
    return -1
  if job_name is None:
    print("Failed to parse job name from output files", file=sys.stderr)
    return -1

  data = gather.output_data(files, all_values)
  if not data:
    return -1

  data_path = datadir / job_name
  if data_path.exists():
    data_path = datadir / "{}-{}".format(job_name, jobid)
    print("Default data file exists. Writing to {} instead".format(data_path))

  with data_path.open("w") as datafile:
    gather.write_data(data, datafile)


if __name__ == "__main__":
  main(sys.argv)
