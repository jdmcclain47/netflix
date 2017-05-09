#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy.distutils import intelccompiler

import os

import numpy

# FAST : the user will need to tune his or her libraries

# Other options include msse4 instead of march=native -fast -funroll-loops
# WARNING: CPU specific options like -march=native will mess up if you are
#          switching between difference architectures on your build and
#          run node (Illegal Instruction)

#Options = "-unroll4 -mtune=native -march=native"
Options = "-unroll4 -march=core-avx2"
os.environ["CC"] = "icc " + Options
os.environ["LDSHARED"] = "icc -shared " + Options
os.environ["CFLAGS"] = "-openmp"
setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("sgd_py",
                             sources=["sgd_py.pyx", "sgd_c.c"],
                             libraries = ['mkl_rt', 'pthread', 'irc'],
                             library_dirs=['/opt/intel/composer_xe_2013_sp1.4.211/mkl/lib/intel64','/opt/intel/composer_xe_2013_sp1.4.211/compiler/lib/intel64',
                                 '/usr/local/intel/lib64/openmpi','/usr/local/openmpi/1.6.5/intel140/x86_64/lib64/'],
                             include_dirs=['/opt/intel/composer_xe_2013_sp1.4.211/mkl/include/',numpy.get_include()])],
)

# SLOW : sometimes slow and steady wins the race

#setup(
#    cmdclass = {'build_ext': build_ext},
#    ext_modules = [Extension("sgd_py",
#                             sources=["sgd_py.pyx", "sgd_c.c"],
#                             include_dirs=[numpy.get_include()])],
#)
