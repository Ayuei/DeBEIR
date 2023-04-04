#!/bin/sh

n=5


echo "Testing compile"
for _ in $(seq 1 ${n}); do
  rm -r cache/
  python compiled_pytorch.py
done

echo "Testing no compile"
for _ in $(seq 1 ${n}); do
  rm -r cache/
  python warm_start_cache.py
done