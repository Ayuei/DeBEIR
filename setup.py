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
    install_requires=['torch>=2.0.0', 'elasticsearch[async]>=8.3.1', 'plac', 'sentence-transformers',
                      'spacy', 'scispacy', 'tqdm', 'pandas', 'trectools', 'numpy',
                      'toml', 'dill', 'requests', 'loguru',
                      'analysis_tools_ir @ git+https://github.com/Ayuei/analysis_tools_ir.git',
                      'allRank @ git+https://github.com/Ayuei/allRank.git',
                      'transformers', 'shutup>=0.2.0', 'datasets', 'scikit-learn',
                      'scipy', 'optuna', 'wandb', 'torch_optimizer'],
)
