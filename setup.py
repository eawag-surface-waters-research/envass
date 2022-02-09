from setuptools import setup, find_packages

with open('README.md') as f:
    README = f.read()

VERSION = '0.0.3'
DESCRIPTION = 'ENVASS ENVironmental data quality ASSurance.'

setup(
    name="envass",
    version=VERSION,
    author="James Runnalls",
    author_email="<james.runnalls@eawag.ch>",
    description=DESCRIPTION,
    long_description=README,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    url='https://github.com/eawag-surface-waters-research/envass',
    install_requires=[],
    license="MIT",
    keywords=['python', 'ENVASS', 'quality assurance', 'environmental data'],
)