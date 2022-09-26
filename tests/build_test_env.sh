#!/bin/sh

git clone git@github.com:Ayuei/go-clinical-indexer.git

echo "Extracting test data"
tar -xf test.tar.gz

docker build -t "debir/test:0.1" .
docker run -d -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -e "xpack.security.enabled=false" -e "xpack.security.http.ssl.enabled=false" --name elasticsearch_debir_test elasticsearch:8.4.1
docker run --rm --network host --name indexer_test_elasticsearch debir/test:0.1

cd ..

python ./examples/indexing/create_semantic_index.py -c tests/config.toml -n tests/nir_config.toml --fields Text -d 100_000 -w 2 -b 100
