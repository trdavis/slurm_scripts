"""Submits an array job and a dependent job that processes its output.

Submits a supplied array job script (presumed to be created with
make_parameter_array_job.py). Also submits a processing job with dependency
on the array job, so it will only run after all jobs in the array finish. The
processing job is given the job number of the array job and the directory it
output to, allowing it to process all of the outputs.

For an example processing script, see gather_array_job_output.py

Requirements: Python3.6+, absl-py

    Example usage:

    python process_array_job.py --jobscript=my_job.sh --process=my_script.py
"""

import re
import subprocess
import sys
import pathlib

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("jobscript", None, "Script for array job")
flags.mark_flag_as_required("jobscript")
flags.DEFINE_string("process", None, "Executable used to process job results")
flags.mark_flag_as_required("process")
flags.DEFINE_string("outdir",
                    str(pathlib.Path.home()) + "/output/",
                    "Directory where array job output is sent")
flags.DEFINE_string("datadir",
                    str(pathlib.Path.home()) + "/data/",
                    "Directory for processed data files")

jobid_regex = re.compile("[0-9]+$")


def main(argv):
  del argv  # unused
  outdir = pathlib.Path(FLAGS.outdir)
  if not outdir.is_dir():
    print("Output directory doesn't exist")
    return -1
  outdir = outdir.resolve()
  datadir = pathlib.Path(FLAGS.datadir)
  if not datadir.is_dir():
    print("Data directory doesn't exist")
    return -1
  datadir = datadir.resolve()
  process = pathlib.Path(FLAGS.process)
  if not process.is_file():
    print("Process script doesn't exist")
    return -1
  process = process.resolve()
  slurm_output = subprocess.check_output(["sbatch", FLAGS.jobscript])
  match = jobid_regex.search(slurm_output.decode("utf-8"))
  if not match:
    print("Failed to parse jobid from slurm output:",
          slurm_output,
          file=sys.stderr)
    return -1
  jobid = match.group()
  out_path = outdir / f"gather-{jobid}.out"
  # Jobs on CC clusters should be at least 1 hour
  subprocess.run([
      "sbatch", f"--job-name=gather-{jobid}", "--time=1:00:00",
      f"--output={out_path}", f"--dependency=afterok:{jobid}",
      f"--wrap={process} {jobid} {outdir} {datadir}"
  ],
                 check=True)


if __name__ == "__main__":
  app.run(main)
