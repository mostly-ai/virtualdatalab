from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy
from Cython.Build import cythonize
setup(
    name="virtualdatalab",
    version="0.1",
    packages=find_packages(),
    ext_modules=cythonize(Extension(
                "virtualdatalab.cython.cython_metric",
                sources=["virtualdatalab/cython/cython_metric.pyx"],
                include_dirs=[numpy.get_include()]),
            ),
    package_data={'virtualdatalab':['datasets/data/*.csv']},
    zip_safe=False
)