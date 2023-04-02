#!/bin/sh

n=5

echo "Testing no cache"

for _ in $(seq 1 ${n}); do
  rm -r cache/
  python warm_start_cache.py
done

echo "Testing cache"
for _ in $(seq 1 ${n}); do
  python warm_start_cache.py
done
