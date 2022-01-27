from setuptools import setup, find_packages

setup(
    name='pyRandomWalk',
    version='0.0.1',
    packages=find_packages(where='src'),
    install_requires=['numpy', 'pandas', 'little_helpers']
)
