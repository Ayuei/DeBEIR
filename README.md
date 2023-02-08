# DeBEIR

A **De**nse **B**i-**E**ncoder for **I**nformation **R**etrieval library for experimenting and using neural models (with a particular emphasis on bi-encoder models) for end-to-end ranking of documents.

###
Requires python >= 3.10

### Setup and installation
It is recommended to set up a virtual environment and install from source

```bash
python3 -m venv venv
source venv/bin/activate

pip install git+https://github.com/Ayuei/DeBEIR.git

# Sentence Segmentation Model install
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_md-0.5.0.tar.gz
```

### Usage

The library has an emphasis on reproducibility and experimentation. With this in mind, settings are placed into configuration files to be used to build the pipeline. 

```python3
from debeir.interfaces.pipeline import NIRPipeline

p = NIRPipeline.build_from_config(config_fp="./tests/config.toml",
                                  engine="elasticsearch",
                                  nir_config_fp="./tests/nir_config.toml")

results = await p.run_pipeline(cosine_offset=5.0)
```

See ```examples/``` for more use cases and where to get started.

### Documentation

API Documentation for the library with rendered HTML documentation is available at [https://ayuei.github.io/DeBEIR/debeir.html](https://ayuei.github.io/DeBEIR/debeir.html) which is built with the pdoc3 library and is rebuilt with every commit with gh-pages.

Statically compiled documentation (which is updated less frequently) can be found in the top level directory [docs/index.html](docs/index.html).

You can also build this documentation with the pdoc library by executing the following commands:
```
pip install -r requirements-docs.txt

pdoc -o docs/ src/debeir/
```

### Development

If you use to help with development of the library, first verify the tests cases and set up a development environment. 
This will take approximately 30 minutes to complete on a mid-range system.

Requires: Docker and pip installation of requirements-dev.txt packages.

```bash
virtualenv venv

source virtualenv/venv/activate

pip install -r requirements-dev.txt

cd tests/

./build_test.env.sh

pytest .
```

A helper script for removing the development environment is provided in ```tests/cleanup.sh```

## Community Guidelines

### An Issue?
If you have any issue with the current library, please file an issue [create an issue](https://github.com/Ayuei/DeBEIR/issues/new/choose).

### Contributing
For those wanting to contribute to the library, please see [CONTRIBUTING.md](CONTRIBUTING.md) and submit a pull request!

### Support
If you wish to reach out to the author and maintainer of this library, please email vincent.nguyen@anu.edu.au
