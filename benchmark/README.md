# Benchmarking

This library contains code to benchmark our API async code against synchronous query calls.

To use the benchmark, simply call:

```bash
./run_all.sh
```

This will create an embedding cache for queries on start-up, and run 3 iterations of each benchmark.

For a more realistic scenario, we performed the benchmark across a network, where the index is located separately from
the query caller.

Our results on our machine can be found here:

|       | Mean  | Stdev | Max   | Min   |
|-------|-------|-------|-------|-------|
| Async | 27.87 | 0.16  | 28.01 | 27.69 |
| Sync  | 212.5 | 2.31  | 215.1 | 210.8 |

We can see that there is roughly a 9-10x speedup. 