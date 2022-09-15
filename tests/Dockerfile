FROM golang

COPY test_set/test.tsv test.tsv
COPY go-clinical-indexer/ indexer/

RUN cd indexer/ && go build .
# Sleep for 30 seconds for the es instance to start.
CMD sleep 30 && ./indexer/indexer -index "test" -collection marco -data_path "test.tsv" -accurate -delete
