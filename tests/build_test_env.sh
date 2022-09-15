#!/bin/sh

git clone git@github.com:Ayuei/go-clinical-indexer.git

tar -xf test.tar.gz

docker build -t "debir/test:0.1" .
docker run --rm -d --network host -e "discovery.type=single-node" -e "xpack.security.enabled=false" -e "xpack.security.http.ssl.enabled=false" --name elasticsearch_debir_test elasticsearch:8.4.1
docker run --rm --network host --name indexer_test_elasticsearch debir/test:0.1
