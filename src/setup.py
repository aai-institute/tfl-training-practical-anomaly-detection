# -*- coding: utf-8 -*-
from setuptools import setup

setup(
    name='src',
    version='0.1',
    url='https://gitlab.aai.lab/workshops/evt',
    license='',
    author='Faried Abu Zaid, Summan Sohail',
    author_email='faried.abuzaid@unternehmertum.de, sohail@unternehmertum.de',
    description='Auxilliary Functions and uni-variate POT method for the Anomaly detection Workshop Exercises',
    py_modules=['exercise_tools', 'vae'],
    install_requires=['scipy', 'numpy', 'pandas', 'matplotlib', 'tqdm',
                      'pyod', 'jupyter','tensorflow','tensorflow-probability','scikit-learn']
)
