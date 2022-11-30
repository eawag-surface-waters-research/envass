from setuptools import setup, find_packages

with open('README.md') as f:
    README = f.read()

VERSION = '0.1.0'
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
    install_requires=[
        'ipywidgets>=7.6.5',
        'numpy>=1.19.5',
        'pandas>=1.1.5',
        'plotly>=5.6.0',
        'scikit-learn>=0.24.2',
        'scipy>=1.5.4'
    ],
    license="MIT",
    keywords=['python', 'ENVASS', 'quality assurance', 'environmental data'],
)