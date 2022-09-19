#!/bin/sh

docker stop elasticsearch_debir_test
docker rm elasticsearch_debir_test
docker rm indexer_test_elasticsearch
rm -rf go-clinical-indexer/
rm -r test_set/
