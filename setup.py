from setuptools import setup, find_packages

version = '0.2.1'

with open('README.md') as readme:
    long_desc = readme.read()

setup(name='sssr',
      description='Self-supervised super-resolution',
      author='Shuo Han',
      author_email='shan50@jhu.edu',
      version=version,
      packages=find_packages(),
      license='GPLv3',
      python_requires='>=3.7.10',
      scripts=['scripts/train.py'],
      long_description=long_desc,
      install_requires=[
          'torch>=1.8.1',
          'improc3d',
          'numpy',
          'scipy',
          'nibabel',
          'resize@git+https://github.com/shuohan/resize@0.1.1',
          'ptxl@git+https://gitlab.com/shan-deep-networks/ptxl@0.3.1',
          'sssrlib@git+https://github.com/shuohan/sssrlib@0.3.0'
      ],
      long_description_content_type='text/markdown',
      url='https://github.com/shuohan/sssr',
      classifiers=[
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Operating System :: OS Independent'
      ]
     )
