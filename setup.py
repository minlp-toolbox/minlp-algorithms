"""A repository for MINLP solvers."""

from setuptools import setup, find_packages

setup(
    name="minlp-algorithms",
    version="0.1",
    packages=find_packages(exclude=['tests']),
    install_requires=[
        "numpy",
        "pandas",
        "casadi",
        "scipy",
        "pytz",
        "matplotlib",
        "parameterized",
        "timeout-decorator",
        "tox",
        "colored",
        "seaborn",
        "argcomplete",
    ]
)
