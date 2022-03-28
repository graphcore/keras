# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import os
import subprocess
import shutil
import sys
import tempfile

THIS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))


def exit_with_error(message, exit_code=1):
  print(f"ERROR: {message}")
  exit(exit_code)


def validate_args(args):
  if not os.path.exists(args.tensorflow_wheel_path):
    exit_with_error(f"Specified tensorflow wheel path does not exist: "
                    f"{args.tensorflow_wheel_path}.")
  for env_arg in args.environment:
    if not "=" in env_arg:
      exit_with_error(f"--environment argument must be in the form "
                      f"VARIABLE_NAME=variable_value: {env_arg}")


def generate_activate_poplar_commands():
  commands = []
  tf_poplar_base = os.environ.get("TF_POPLAR_BASE")
  tf_poplar_sandbox = os.environ.get("TF_POPLAR_SANDBOX")
  popc_from_environment = shutil.which("popc")

  def activate_poplar_from_var(var_name, popc_path, activate_path):
    if not os.path.exists(popc_path):
      exit_with_error(
          f"{var_name} is set to an invalid path: '{popc_path}' does not exist."
      )
    if popc_from_environment is None:
      commands.append(f"source {activate_path}")
    elif not os.path.samefile(popc_path, popc_from_environment):
      exit_with_error(f"{var_name} is set to {os.environ[var_name]} but a "
                      f"different Poplar installation has been activated "
                      f"({popc_from_environment}). Please unset {var_name} or "
                      f"deactivate the other Poplar installation.")
    print(f"Using Poplar from {var_name}: '{os.environ[var_name]}'.")

  if tf_poplar_base is not None and tf_poplar_sandbox is not None:
    exit_with_error(
        "Both TF_POPLAR_BASE and TF_POPLAR_SANDBOX are set, please "
        "unset one.")

  if tf_poplar_base is not None:
    popc_path = os.path.join(tf_poplar_base, "bin/popc")
    enable_script = os.path.join(tf_poplar_base, "enable.sh")
    activate_poplar_from_var("TF_POPLAR_BASE", popc_path, enable_script)
  elif tf_poplar_sandbox is not None:
    popc_path = os.path.join(tf_poplar_sandbox, "poplar/bin/popc")
    activate_script = os.path.join(tf_poplar_sandbox, "../activate.sh")
    activate_poplar_from_var("TF_POPLAR_SANDBOX", popc_path, activate_script)
  elif popc_from_environment is not None:
    print(f"Using activated Poplar installation: '{popc_from_environment}'")
  else:
    exit_with_error(
        "You need to activate a Poplar installation, or set either "
        "TF_POPLAR_BASE or TF_POPLAR_SANDBOX.")
  return commands


def generate_create_and_source_venv_commands(args):
  commands = []
  if os.path.exists(args.venv_path):
    print(f"Re-using existing virtual environment in '{args.venv_path}'.")
  else:
    print(f"Creating Python virtual environment in '{args.venv_path}'.")
    # Set up an independent python virtualenv.
    commands.append(f"python3 -m venv {args.venv_path} --without-pip")

  # Activate venv and install TensorFlow.
  venv_activate_script = os.path.join(args.venv_path, "bin/activate")
  commands.append(f"source {venv_activate_script}")
  return commands


def generate_install_latest_pip_commands():
  commands = []
  if str(sys.version).startswith("3.6."):
    commands.append(
        "wget -O get-pip.py https://bootstrap.pypa.io/pip/3.6/get-pip.py")
  else:
    commands.append(
        "wget -O get-pip.py https://bootstrap.pypa.io/pip/get-pip.py")
  commands.append("python3 get-pip.py")
  commands.append("rm -f get-pip.py")
  return commands


def generate_install_pip_package_commands(args):
  return [
      "pip3 install --upgrade setuptools wheel",
      f"pip3 install --force-reinstall {args.tensorflow_wheel_path}"
  ]


def parse_env_vars(env_vars):
  environment = {}
  for env_var in env_vars:
    kvp = env_var.split("=", 1)
    if len(kvp) == 2 and all(kvp):
      environment[kvp[0]] = kvp[1]
  return environment


def run_and_get_environment(commands):
  supressed_commands = (f"{command} 1>/dev/null"
                        for command in environment_setup_commands)
  combined_commands = " && ".join(supressed_commands)
  get_env_command = ["env", "-i", "bash", "-c", f"{combined_commands} && env"]

  print("Generating build environment with commands:")
  for command in commands:
    print(f"  {command}")

  # Visually separate command output in the console.
  print()
  process = subprocess.Popen(get_env_command, stdout=subprocess.PIPE)
  try:
    stdout, _ = process.communicate(timeout=60)
    print()

  except TimeoutExpired:
    process.kill()
    exit_with_error("Command to generate build environment timed out.")
  if process.returncode != 0:
    exit_with_error(f"Command to generate build environment exited with code "
                    f"{process.returncode}.")

  environment = parse_env_vars(stdout.decode().split("\n"))
  print("Build environment generated.")
  return environment


def write_user_bazelrc(args, environment):
  use_backup = False
  user_bazelrc_path = None
  try:
    # Backup current user config if it exists.
    user_bazelrc_path = os.path.join(THIS_DIR, ".bazelrc.user")
    use_backup = os.path.exists(user_bazelrc_path)
    if use_backup:
      tempdir = tempfile.mkdtemp()
      backup_user_bazelrc_path = os.path.join(tempdir, ".bazelrc.user")
      shutil.copy(user_bazelrc_path, backup_user_bazelrc_path)

    # Paths pointing to venv, Poplar, and TensorFlow.
    with open(".bazelrc.user", "w") as user_bazelrc:
      user_bazelrc.write(f"build --action_env=PATH='{environment['PATH']}'\n")
      user_bazelrc.write(f"build --action_env=LD_LIBRARY_PATH='{environment['LD_LIBRARY_PATH']}'\n")  # yapf: disable
      user_bazelrc.write(f"build --action_env=PYTHONPATH='{environment['PYTHONPATH']}'\n")  # yapf: disable

      if "TMPDIR" in environment:
        user_bazelrc.write(f"build --action_env=TMPDIR='{environment['TMPDIR']}'\n")  # yapf: disable
      if args.environment:
        for k, v in parse_env_vars(args.environment).items():
          user_bazelrc.write(f"build --action_env={k}='{v}'\n")

      if args.disk_cache is not None:
        user_bazelrc.write(f"build --disk_cache='{args.disk_cache}'\n")

      # Restore the user defined options.
      separation = "### ADD USER DEFINED OPTIONS AFTER THIS LINE ONLY ###"
      user_bazelrc.write(f"{separation}\n")
      if use_backup:
        with open(backup_user_bazelrc_path, "r") as backup_user_bazelrc:
          text = backup_user_bazelrc.read()
          offset = text.find(separation)
          if offset != -1:
            offset += len(separation)
            user_bazelrc.write(text[offset:])

      print("Successfully created .bazelrc.user")
  except Exception as e:  # pylint: disable=broad-except
    # Delete incomplete user bazelrc.
    shutil.rmtree(user_bazelrc_path, ignore_errors=True)
    if use_backup:
      shutil.copy(backup_user_bazelrc_path, user_bazelrc_path)
    exit_with_error(f"Failed to create .bazelrc.user: {e}")
  finally:
    if use_backup:
      shutil.rmtree(tempdir, ignore_errors=True)


def main():
  # Formatter adds default values to the --help message.
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      'tensorflow_wheel_path',
      help=
      "The path to the IPU TensorFlow .whl file you want to build Keras for.")
  parser.add_argument(
      '--venv-path',
      dest='venv_path',
      default=os.path.join(THIS_DIR, "venv"),
      help="Choose the location of the Python virtual environment. "
      "If a virtual environment already exists at the specified path, it will "
      "be reused.")
  parser.add_argument(
      '--no-user-bazelrc',
      dest='user_bazelrc',
      action="store_false",
      default=True,
      help="Do not touch .bazelrc.user. "
      "Important: This means the environment will not be frozen.")
  parser.add_argument(
      '--environment',
      dest='environment',
      action='append',
      default=[],
      help="Specify an environment variable to forward to Bazel. "
      "The variables will be set during builds and testing. "
      "This argument can be passed multiple times to specify multiple "
      "environment variables.")
  parser.add_argument(
      '-c',
      '--disk-cache',
      dest='disk_cache',
      default=None,
      help="Enable and specify the directory for Bazel's disk cache.")

  args = parser.parse_args()
  validate_args(args)

  environment_setup_commands = (
      generate_activate_poplar_commands() +
      generate_create_and_source_venv_commands(args) +
      generate_install_latest_pip_commands() +
      generate_install_pip_package_commands(args))

  environment = run_and_get_environment(environment_setup_commands)

  if args.user_bazelrc:
    write_user_bazelrc(args, environment)

  print("Configure script completed successfully!")


if __name__ == '__main__':
  main()
