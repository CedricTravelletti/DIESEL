from setuptools import setup
from setuptools import find_packages
from distutils.cmd import Command
from distutils.extension import Extension
import os
import sys
import io
import subprocess
import platform


# Make sure numpy and Cython get installed first.
from setuptools import dist
dist.Distribution().fetch_build_eggs(['Cython>=0.29.15', 'numpy>=1.18.0'])

import numpy as np
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

if "--line_trace" in sys.argv:
    line_trace = True
    print("Build with line trace enabled ...")
    sys.argv.remove("--line_trace")
else:
    line_trace = False

PACKAGE = "diesel"
NAME = "DIESEL"
VERSION = "0.0.1"
DESCRIPTION = "DIstributed EStimation of EnsembLe Covariance " + VERSION
AUTHOR = "Cedric Travelletti"
AUTHOR_EMAIL = "cedrictravelletti@gmail.com"
URL = 'https://github.com/CedricTravelletti/DIESEL'


requirements = "requirements.txt"

ext_modules = [
    # "climate/distances.pyx",
    # "volcapy/kernels/exponential.pyx",
    # "volcapy/niklas/banerjee.pyx",
    # "volcapy/niklas/banerjee2d.pyx",
    # "volcapy/grid/covariance_tools.pyx",
]


def generate_extensions(ext_modules, line_trace=False):

    extensions = []

    if line_trace:
        print("define cython trace to True ...")
        define_macros = [('CYTHON_TRACE', 1), ('CYTHON_TRACE_NOGIL', 1)]
    else:
        define_macros = []

    for pyxfile in ext_modules:
        ext = Extension(name='.'.join(pyxfile.split('/'))[:-4],
                        sources=[pyxfile],
                        define_macros=define_macros)
        extensions.append(ext)
    return extensions


n_cpu = 4
ext_modules_settings = cythonize(
        generate_extensions(ext_modules, line_trace), 
        compiler_directives={'embedsignature': True, 'linetrace': line_trace}, nthreads=n_cpu)


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    packages=find_packages(),
    include_package_data=False,
    install_requires=[# io.open(requirements, encoding='utf8').read(),
            # 'mvnorm @ git+https://github.com/CedricTravelletti/torch-mvnorm.git#egg=mvnorm'
            ],
    classifiers=[],
    ext_modules=ext_modules_settings,
    include_dirs=[np.get_include()],
)
