from setuptools import setup, find_packages

VERSION = '0.0.3'
DESCRIPTION = 'ENVASS ENVironmental data quality ASSurance.'
LONG_DESCRIPTION = 'ENVironmental data quality ASSurance for generating high quality data products.'

setup(
    name="envass",
    version=VERSION,
    author="James Runnalls",
    author_email="<runnalls.james@gmail.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],
    license="MIT",
    keywords=['python', 'ENVASS', 'quality assurance', 'environmental data'],
)
