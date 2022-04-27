# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Utils to help build and verify pip package for Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import fnmatch
import os

PIP_EXCLUDED_FILES = frozenset([
    'keras/api/create_python_api_wrapper.py',
    'keras/applications/efficientnet_weight_update_util.py',
    'keras/distribute/tpu_strategy_test_utils.py',
    'keras/tools/pip_package/setup.py',
    'keras/tools/pip_package/create_pip_helper.py',
])

PIP_EXCLUDED_DIRS = frozenset([
    'keras/benchmarks',
    'keras/integration_tests',
    'keras/tests',
    # Begin IPU specific changes.
    'keras/ipu/docs',
    # Enf IPU specific changes.
])

# Directories that should not have __init__.py files generated within them.
EXCLUDED_INIT_FILE_DIRECTORIES = frozenset([
    'keras/benchmarks',
    'keras/tools',
])


class PipPackagingError(Exception):
  pass


def create_init_files(pip_root):
  """Create __init__.py in pip directory tree.

  These files are auto-generated by Bazel when doing typical build/test, but
  do not get auto-generated by the pip build process. Currently, the entire
  directory tree is just python files, so its fine to just create all of the
  init files.

  Args:
    pip_root: Root directory of code being packaged into pip.
  """
  for path, subdirs, _ in os.walk(pip_root):
    for subdir in subdirs:
      init_file_path = os.path.join(path, subdir, '__init__.py')
      if any(excluded_path in init_file_path
             for excluded_path in EXCLUDED_INIT_FILE_DIRECTORIES):
        continue
      if not os.path.exists(init_file_path):
        # Create empty file
        open(init_file_path, 'w').close()


def verify_python_files_in_pip(pip_root, bazel_root):
  """Verifies all expected files are packaged into Pip.

  Args:
    pip_root: Root directory of code being packaged into pip.
    bazel_root: Root directory of Keras Bazel workspace.

  Raises:
    PipPackagingError: Missing file in pip.
  """
  for path, _, files in os.walk(bazel_root):
    if any(d for d in PIP_EXCLUDED_DIRS if d in path):
      # Skip any directories that are exclude from PIP, eg tests.
      continue

    python_files = set(fnmatch.filter(files, '*.py'))
    python_test_files = set(fnmatch.filter(files, '*test.py'))
    python_benchmark_files = set(fnmatch.filter(files, '*benchmark.py'))
    # We only care about python files in the pip package, see create_init_files.
    files = python_files - python_test_files - python_benchmark_files
    for f in files:
      pip_path = os.path.join(pip_root, os.path.relpath(path, bazel_root), f)
      file_name = os.path.join(path, f)
      path_exists = os.path.exists(pip_path)
      file_excluded = file_name.lstrip('./') in PIP_EXCLUDED_FILES
      if not path_exists and not file_excluded:
        raise PipPackagingError(
            ('Pip package missing the file %s. If this is expected, add it '
             'to PIP_EXCLUDED_FILES in create_pip_helper.py. Otherwise, '
             'make sure it is a build dependency of the pip package') %
            file_name)
      if path_exists and file_excluded:
        raise PipPackagingError(
            ('File in PIP_EXCLUDED_FILES included in pip. %s' % file_name))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--bazel-root',
      type=str,
      required=True,
      help='Root directory of Keras Bazel workspace.')
  parser.add_argument(
      '--pip-root',
      type=str,
      required=True,
      help='Root directory of code being packaged into pip.')

  args = parser.parse_args()
  create_init_files(args.pip_root)
  verify_python_files_in_pip(args.pip_root, args.bazel_root)


if __name__ == '__main__':
  main()
