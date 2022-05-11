import plac
from elasticsearch import AsyncElasticsearch
from executor.runner import ClinicalTrialsExecutor
from parsers.query_topic_parsers import CDS2021Parser
from query.query import TrialsQuery
from query_builder.embeddings import Encoder
import asyncio


@plac.opt("addr", "Elasticsearch Address", type=str)
@plac.opt("es_port", "Elasticsearch Port Number", type=int)
@plac.opt("topics_path", "Path to query topics", type=str)
@plac.opt("model_path", "Path to Sentence Encoder model", type=str)
@plac.opt("query_type", "Query type", type=str)
@plac.opt("norm_weight", "Norm Weight", type=float)
@plac.opt("index_name", "Name of Elasticsearch Index", type=str)
@plac.opt("gmapping_id", "mapping id", type=int)
@plac.opt("output_file", "Output file name and/or path", type=str)
@plac.flg("delete", "Overwrite output file it exists")
@plac.opt("size", "Retrieved Input Size", type=int)
def main(
    addr,
    es_port,
    topics_path,
    model_path,
    query_type,
    index_name,
    output_file,
    gmapping_id=None,
    norm_weight=2.15,
    size=1000,
    delete=False,
):
    topics = CDS2021Parser.get_topics(csvfile=open(topics_path))
    es = AsyncElasticsearch([{"host": addr, "port": es_port}], timeout=1800)
    query = TrialsQuery(
        topics=topics, mappings_path="./assets/mapping.json", query_type="ablation"
    )
    encoder = Encoder(model_path)

    if gmapping_id:
        query.fields = [gmapping_id - 1]
        print(query.mappings[gmapping_id - 1])
        output_file = output_file + "_" + str(gmapping_id - 1)

    if delete:
        open(output_file, "w+").close()

    assert query_type in ["ablation", "query", "embedding"]

    ex = ClinicalTrialsExecutor(
        topics=topics,
        client=es,
        index_name=index_name,
        output_file=output_file,
        return_size=size,
        query=query,
        encoder=encoder,
    )

    asyncio.run(
        ex.run_all_queries(
            serialize=True,
            query_type=query_type,
            norm_weight=norm_weight,
            ablations=True,
        )
    )

    asyncio.run(es.close())


if __name__ == "__main__":
    plac.call(main)
