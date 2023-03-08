import threading
from pathlib import Path
from queue import Queue

import plac
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

from debeir.core.config import GenericConfig, _NIRMasterConfig
from debeir.core.indexer import SemanticElasticsearchIndexer
from debeir.datasets.factory import config_factory
from debeir.rankers.transformer_sent_encoder import Encoder


class ProducerThread(threading.Thread):
    def __init__(self, client, index, queue: Queue, total_docs=None):
        super().__init__()
        self.client = client
        self.index = index
        self.q = queue
        self.total_docs = total_docs

    def run(self):
        for document in tqdm(helpers.scan(self.client, index=self.index), total=self.total_docs):
            self.q.put(document)

        return


@plac.pos('config', 'Path of configuration file', type=Path)
@plac.pos('nir_config', 'Path of master nir configuration file', type=Path)
@plac.opt('fields', 'Fields to encode. Format like so: field1,field2,field3', type=str)
@plac.opt('documents', 'Number of documents. This is used for progress bar.', type=int)
@plac.opt('buffer_size', 'Queue document buffer size', type=int)
@plac.opt('workers', 'Number of workers for indeixng', type=int)
def main(config, nir_config, fields, documents=None,
         buffer_size=100, workers=2):
    q = Queue(buffer_size)

    config = config_factory(config, GenericConfig)

    es_config = config_factory(
        nir_config, _NIRMasterConfig
    ).get_search_engine_settings(return_as_instance=True)

    es_client = Elasticsearch(f"{es_config.protocol}://{es_config.ip}:{es_config.port}",
                              request_timeout=es_config.timeout)
    es_client_thread = Elasticsearch(f"{es_config.protocol}://{es_config.ip}:{es_config.port}",
                                     request_timeout=es_config.timeout)

    p = ProducerThread(es_client,
                       index=config.index,
                       queue=q,
                       total_docs=documents)

    p.start()

    fields_to_encode = fields.split(',')

    threads = []

    for _ in range(workers):
        indexer_t = SemanticElasticsearchIndexer(es_client_thread,
                                                 encoder=Encoder(config.encoder_fp),
                                                 index=config.index,
                                                 fields_to_encode=fields_to_encode,
                                                 queue=q)
        threads.append(indexer_t)
        indexer_t.start()

    p.join()

    for t in threads:
        t.join()


if __name__ == "__main__":
    try:
        plac.call(main)
    except Exception as e:
        raise e
