#!/usr/bin/env python

from setuptools import setup, find_packages
import setuptools.command.develop
import setuptools.command.build_py
import os
import subprocess
from os.path import exists

version = '0.0.1'

# Adapted from https://github.com/pytorch/pytorch
cwd = os.path.dirname(os.path.abspath(__file__))
if os.getenv('FFTNET_PYTORCH_BUILD_VERSION'):
    version = os.getenv('FFTNET_PYTORCH_BUILD_VERSION')
else:
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
        version += '+' + sha[:7]
    except subprocess.CalledProcessError:
        pass
    except IOError:  # FileNotFoundError for python 3
        pass


class build_py(setuptools.command.build_py.build_py):

    def run(self):
        self.create_version_file()
        setuptools.command.build_py.build_py.run(self)

    @staticmethod
    def create_version_file():
        global version, cwd
        print('-- Building version ' + version)
        version_path = os.path.join(cwd, 'version.py')
        with open(version_path, 'w') as f:
            f.write("__version__ = '{}'\n".format(version))


class develop(setuptools.command.develop.develop):

    def run(self):
        build_py.create_version_file()
        setuptools.command.develop.develop.run(self)


def create_readme_rst():
    global cwd
    try:
        subprocess.check_call(
            ["pandoc", "--from=markdown", "--to=rst", "--output=README.rst",
             "README.md"], cwd=cwd)
        print("Generated README.rst from README.md using pandoc.")
    except subprocess.CalledProcessError:
        pass
    except OSError:
        pass


setup(name='TTS',
      version=version,
      url='https://github.com/mozilla/FFTNet',
      description='FFTNet vocoder for speech synthesis.',
      packages=find_packages(),
      cmdclass={
          'build_py': build_py,
          'develop': develop,
      },
      install_requires=[
          "numpy",
          "scipy",
          "librosa",
          "torch >= 0.4.0",
      ],
      extras_require={
          "bin": [
              "tqdm",
              "tensorboardX",
              "tensorboard",
              "requests",
          ],
      })
