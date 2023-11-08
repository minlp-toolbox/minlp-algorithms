"""A repository for MINLP solvers."""

from setuptools import setup, find_packages

setup(
    name="benders-exp",
    version="0.1",
    packages=find_packages(exclude=['tests']),
    install_requires=[
        "numpy",
        "pandas",
        "casadi",
        "scipy",
        "gurobipy",
        "pytz",
        "matplotlib",
        "parameterized",
        "timeout-decorator",
        "tox",
        "colored"
    ]
)
