from setuptools import setup, find_packages
import os

_base = os.path.dirname(os.path.abspath(__file__))
_requirements = os.path.join(_base, 'requirements.txt')

version = '1.0.0'

install_requirements = []
with open(_requirements) as f:
    install_requirements = f.read().splitlines()


setup(name='gold_biome_classifier',
      version=version,
      description='Utility to predict GOLD biomes from a free text field',
      author='Maxim Scheremetjew, Miguel Boland',
      url='https://github.com/EBI-Metagenomics/biome_prediction',
      packages=find_packages(),
      install_requires=install_requirements,
      entry_points={
          'console_scripts': [
              'predict-biome=load_classifier:main',
          ]
      },
      setup_requires=['pytest-runner'],
      )
