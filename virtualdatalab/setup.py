from setuptools import setup, find_packages
setup(
    name="virtualdatalab",
    version="0.1",
    packages=find_packages(),
    package_data={'virtualdatalab':['datasets/data/*.csv']},
)