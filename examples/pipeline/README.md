## Pipeline

### Prequisties

- Requires a working **Elasticsearch** instance. Point to the correct port and url in ```nir.toml```. To setup an test
  index, see ```./DeBEIR/tests/build_test_env.sh```.
- DeBEIR is installed; which requires
- Python >= 3.10

### Running

To run the example, simply call it using python after setting the appropriate variables in ```config.toml```
and ```nir.toml```.

```python3
# Run the example in the same directory
# We use the test files included in the ./DeBEIR/test/ directory in the script
python3
run_pipeline.py
```
