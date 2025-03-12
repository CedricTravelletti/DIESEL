import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

extensions = [Extension(name="diesel.utils.haversine", sources=["diesel/utils/haversine.pyx"])]

setup(
    name="diesel",
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
)
