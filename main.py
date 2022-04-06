import os
import sys

import plac
from elasticsearch import AsyncElasticsearch
from utils.runner import ClinicalTrialsExecutor
from utils.parsers import CDS2021Parser
from utils.query import TrialsQuery
from utils.embeddings import Encoder
from utils.scaler import unpack_scores
from utils.factory import query_config_factory
import asyncio


@plac.opt('address', "Elasticsearch Address", type=str)
@plac.opt('es_port', "Elasticsearch Port Number", type=int)
@plac.opt('topics_path', "Path to query topics", type=str)
@plac.opt('model_path', "Path to Sentence Encoder model", type=str)
@plac.opt('query_type', "Query type", type=str)
@plac.opt('norm_weight', "Norm Weight", type=str)
@plac.opt('index_name', "Name of Elasticsearch Index", type=str)
@plac.opt('output_file', "Output file name and/or path", type=str)
@plac.opt('config_path', "Path to Run Config File", type=str)
@plac.flg('delete', "Overwrite output file it exists")
@plac.opt('size', "Retrieved Input Size", type=int)
def main(address, es_port, topics_path, output_file=None, index_name=None, query_type=None, config_path=None, model_path=None,
         norm_weight="2.15", size=1000, delete=False):

    loop = asyncio.get_event_loop()
    topics = CDS2021Parser.get_topics(csvfile=open(topics_path))
    es = AsyncElasticsearch([{'host': address, 'port': es_port}], timeout=1800)

    query = None
    encoder = None
    config = None

    if output_file is None:
        os.makedirs(name=f"outputs/{index_name}", exist_ok=True)
        output_file = f"outputs/{index_name}/{config_path.split('/')[-1].replace('.toml', '')}"
        print(output_file)

    if os.path.exists(output_file) and not delete:
        print("Output file exists, skipping")
        sys.exit(0)

    if config_path:
        query, config = query_config_factory(topics, config_path, index_name)
        assert query_type or config.query_type, "At least config or argument must be provided for query type"
        index_name = config.index if config.index else index_name
        assert index_name is not None, "Must provide an index name somewhere"
    else:
        query = TrialsQuery(query_type=query_type, topics=topics, mappings_path="./assets/mapping.json")

    if model_path:
        encoder = Encoder(model_path)

    if norm_weight != "automatic":
        norm_weight = float(norm_weight)

    if delete:
        open(output_file, "w+").close()

    results = None

    ex = ClinicalTrialsExecutor(topics=topics, client=es,
                                index_name=index_name, output_file=output_file,
                                return_size=size, query=query, encoder=encoder, config=config)

    if norm_weight == "automatic" or config.automatic is True:
        print("Running trial queries to get automatic weight adjustment")
        ex.return_size = 1
        ex.return_id_only = True

        temp = config.query_type
        config.query_type = "query"

        results = loop.run_until_complete(ex.run_all_queries(serialize=False, return_results=True))
        results = unpack_scores(results)
        ex.return_size = size
        config.query_type = temp

    loop.run_until_complete(ex.run_all_queries(serialize=True, query_type=query_type, norm_weight=norm_weight,
                                               automatic_scores=results))
    loop.run_until_complete(es.close())
    loop.close()


if __name__ == "__main__":
    plac.call(main)
