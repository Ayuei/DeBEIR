from setuptools import find_packages, setup

setup(
    name='DeBeIR',
    version='0.1.1',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    url='',
    license='GNU GPLv3',
    author='Vincent Nguyen',
    author_email='vincent.nguyen@anu.edu.au',
    description='Dense Bi-encoders for end-to-end Dense Information Retrieval',
    install_requires=['torch>=2.0.0', 'elasticsearch[async]>=8.3.1', 'plac', 'sentence-transformers>=2.2.2',
                      'spacy~=3.2.4', 'scispacy', 'tqdm~=4.64.0', 'pandas', 'trectools', 'numpy~=1.22.4',
                      'toml>=0.10.2', 'dill~=0.3.5.1', 'requests>=2.28.0', 'loguru>=0.6.0',
                      'analysis_tools_ir @ git+https://github.com/Ayuei/analysis_tools_ir.git',
                      'allRank @ git+https://github.com/Ayuei/allRank.git',
                      'transformers>=4.22.0', 'shutup>=0.2.0', 'datasets>=2.4.0', 'scikit-learn>=1.1.2',
                      'scipy', 'optuna>=3.0.2', 'wandb>=0.13.3', 'torch_optimizer'],
)
