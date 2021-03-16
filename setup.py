from setuptools import setup
from setuptools.extension import Extension
import numpy
import versioneer
from Cython.Build import cythonize

ext_modules = cythonize(
    Extension(
        "virtualdatalab.cython.cython_metric",
        sources=["virtualdatalab/cython/cython_metric.pyx"],
        include_dirs=[numpy.get_include()],
    )
)

setup(
    version=versioneer.get_version(),
    ext_modules=ext_modules,
    package_data={"virtualdatalab": ["datasets/data/*"]},
)
