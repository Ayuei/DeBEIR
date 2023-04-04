# Benchmarking

## Query calls
This library contains code to benchmark our API async code against synchronous query calls.

To use the benchmark, simply call:

```bash
./run_all.sh
```

This will create an embedding cache for queries on start-up, and run 5 iterations of each benchmark.

For a more realistic scenario, we performed the benchmark across a network, where the index is located separately from
the query caller.

Our querying results on our machine can be found here:

|       | Mean (s) | Stdev | Max   | Min   |
|-------|----------|-------|-------|-------|
| Async | 27.87    | 0.16  | 28.01 | 27.69 |
| Sync  | 212.5    | 2.31  | 215.1 | 210.8 |

We can see that there is roughly a 9-10x speedup.

## Caching

We also demonstrate the speed of caching the expensive query embedding calls, which is useful in a research setting as
often experiments are repeated several times.

|          | Mean (s) | Stdev  | Max   | Min   |
|----------|----------|--------|-------|-------|
| Cache    | 0.178    | 0.0049 | 0.181 | 0.169 |
| No cache | 28.52    | 0.7664 | 29.03 | 27.19 |

## Torch compile

We also have a benchmark for testing the pytorch 2.0.0 compile performance gains: ```benchmark_pytorch_compile.sh```
We don't see a noticeable difference, which shows that during the encode stage that calls to the GPU model is not the
bottleneck.
Most notably, it is the sentence segmentation that takes the longest.

|                  | Mean (s) | Stdev | Max   | Min   |
|------------------|----------|-------|-------|-------|
| Pytorch Compiled | 28.47    | 0.786 | 29.65 | 27.29 |
| Pytorch          | 28.29    | 0.473 | 29.10 | 27.72 |