#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import, print_function

import io
import sys
from os.path import dirname, join

import versioneer
from setuptools import find_packages, setup

# Give setuptools a hint to complain if it's too old a version
# 24.2.0 added the python_requires option
# Should match pyproject.toml
SETUP_REQUIRES = ['setuptools >= 24.2.0']
# This enables setuptools to install wheel on-the-fly
SETUP_REQUIRES += ['wheel'] if 'bdist_wheel' in sys.argv else []


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


opts = dict(
    name='cucim',
    version=read('VERSION').strip(),  # versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='Apache-2.0',
    description='cuCIM - an extensible toolkit designed to provide GPU accelerated I/O, computer vision & image processing primitives for N-Dimensional images with a focus on biomedical imaging.',  # noqa
    long_description='%s\n%s' % (
        read('README.md'),
        read('CHANGELOG.md')
    ),
    long_description_content_type='text/markdown',
    author='NVIDIA Corporation',
    url='https://github.com/rapidsai/cucim',
    packages=find_packages('src'),
    package_dir={'cucim': 'src/cucim'},
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list:
        #   http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Healthcare Industry',
        'Operating System :: POSIX :: Linux',
        'Environment :: Console',
        'Environment :: GPU :: NVIDIA CUDA :: 11.0',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: C++',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        # 'Operating System :: OS Independent',
        # 'Operating System :: Unix',
        # 'Operating System :: POSIX',
        # 'Operating System :: Microsoft :: Windows',
        # 'Programming Language :: Python :: Implementation :: CPython',
        # uncomment if you test on these interpreters:
        # 'Programming Language :: Python :: Implementation :: PyPy',
        # 'Programming Language :: Python :: Implementation :: IronPython',
        # 'Programming Language :: Python :: Implementation :: Jython',
        # 'Programming Language :: Python :: Implementation :: Stackless',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    project_urls={
        'Documentation': 'https://cucim.readthedocs.io/',
        'Changelog': 'https://cucim.readthedocs.io/en/latest/changelog.html',
        'Issue Tracker': 'https://github.com/rapidsai/cucim/issues',
    },
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    python_requires='>= 3.6',
    platforms=['manylinux2014_x86_64'],
    setup_requires=SETUP_REQUIRES,
    install_requires=[
        # TODO: Check cupy dependency based on cuda version
        'click', 'numpy',  # 'scipy', 'scikit-image'
        # eg: 'aspectlib==1.1.1', 'six>=1.7',
    ],
    extras_require={
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
    entry_points={
        'console_scripts': [
            'cucim = cucim.clara.cli:main',
        ]
    },
)

if __name__ == '__main__':
    setup(**opts)
