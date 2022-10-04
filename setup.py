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
    description='Dense Bi-encoders for end-to-end Dense Information Retrieval',
    install_requires=['torch', 'elasticsearch[async]==8.3.1', 'plac', 'sentence-transformers==2.2.2', 'spacy~=3.2.4',
                      'scispacy',
                      "en_core_sci_md @ https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_md-0.5.0.tar.gz",
                      "analysis_tools_ir",
                      'tqdm~=4.64.0', 'pandas', 'trectools', 'numpy~=1.22.4', 'toml~=0.10.2',
                      'dill~=0.3.5.1', 'requests~=2.28.0', 'loguru~=0.6.0',
                      'allRank @ git+https://github.com/Ayuei/allRank.git',
                      'transformers==4.22.0', 'shutup~=0.2.0', 'datasets==2.4.0', 'jupyterlab==3.4.7',
                      'scikit-learn==1.1.2', 'scipy~=1.9.1', 'optuna==3.0.2', 'wandb==0.13.3', 'torch_optimizer'],
    dependency_link = [
        "https://github.com/Ayuei/analysis_tools_ir/releases/download/v0.0.3/analysis_tools_ir-0.0.3-py3-none-any.whl",
    ]
)
