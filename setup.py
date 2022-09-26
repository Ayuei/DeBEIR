from setuptools import setup, find_packages

setup(
    name='DeBeIR',
    version='0.0.1',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    url='',
    license='GNU GPLv3',
    author='Vincent Nguyen',
    author_email='vincent.nguyen@anu.edu.au',
    description='Dense Bi-encoders for end-to-end Dense Information Retrieval'
)
