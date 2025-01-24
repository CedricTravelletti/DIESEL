from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

# Define the extension module
extensions = [
    Extension("haversine", ["diesel/haversine.pyx"]),
]

# Use cythonize on the extension object.
setup(
    name="DIESEL",
    packages=["diesel"],  # Replace with the names of your packages
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)

