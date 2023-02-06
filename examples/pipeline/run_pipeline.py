"""
Example usage for Pipeline

Build a Pipeline from configuration files and run the pipeline.
"""
import asyncio
import shutil
import tarfile

from debeir import NIRPipeline

if __name__ == "__main__":
    shutil.copy("../../tests/test_set.tar.gz", ".")
    with tarfile.open("./test_set.tar.gz", mode="r") as f:
        f.extractall()

    p = NIRPipeline.build_from_config(config_fp="./config.toml",
                                      engine="elasticsearch",
                                      nir_config_fp="./nir.toml")

    results = asyncio.run(p.run_pipeline())
