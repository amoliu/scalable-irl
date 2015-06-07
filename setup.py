from setuptools import setup
from setuptools.extension import Extension
import numpy as np

import os

if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

setup(name="sirl",
      version="0.0.1",
      install_requires=["numpy", "networkx"],
      packages=['sirl',
                'sirl.algorithms',
                'sirl.domains',
                'sirl.utils',
                'sirl.tests',
                'sirl.tests.test_algorithms',
                'sirl.tests.test_domains',
                'sirl.tests.test_utils'
                ],
      include_package_data=True,
      description="Sample (Graph) Based Inverse Reinforcement Learning",
      author="Billy Okal",
      author_email="sudo@makokal.com",
      url="http://sample-irl.github.io",
      license="New BSD",
      use_2to3=True,
      classifiers=['Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX',
                   'Operating System :: Unix',
                   'Operating System :: MacOS',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.6',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.3',
                   ],
      include_dirs=[np.get_include()]
      )
