import threading
from queue import Queue

from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

from debeir.core.config import ElasticsearchConfig, GenericConfig

BUF_SIZE = 10000
N_THREADS = 6


class ProducerThread(threading.Thread):
    def __init__(self, client, index, queue: Queue):
        super().__init__()
        self.client = client
        self.index = index
        self.q = queue

    def run(self):
        for document in tqdm(helpers.scan(self.client, index=self.index), total=360_000):
            q.put(document)

        return


class ConsumerThread(threading.Thread):
    def __init__(self, client, index, queue: Queue):
        super().__init__()
        self.client = client
        self.index = index
        self.q = queue

    def run(self):
        while not self.q.empty():
            document = self.q.get()
            self.index_document(document)

    def index_document(self, document):
        update_doc = {}
        doc = document["_source"]
        update_doc["docid"] = doc["IDInfo"]["NctID"]

        if update_doc:
            self.client.update(index=self.index,
                               id=document['_id'],
                               doc=update_doc)


if __name__ == "__main__":
    q = Queue(BUF_SIZE)

    config = GenericConfig.from_toml(
        fp="/home/vin/Projects/nir/configs/trec2022/embeddings.toml",
        field_class=GenericConfig
    )

    config.encoder = None
    es_config = ElasticsearchConfig.from_toml("./configs/elasticsearch.toml",
                                              field_class=ElasticsearchConfig)

    es_client = Elasticsearch(f"{es_config.protocol}://{es_config.ip}:{es_config.port}",
                              timeout=es_config.timeout)
    es_client_thread = Elasticsearch(f"{es_config.protocol}://{es_config.ip}:{es_config.port}",
                                     timeout=es_config.timeout)

    p = ProducerThread(es_client,
                       index=config.index,
                       queue=q)

    p.start()
    import time; time.sleep(2)

    threads = [ConsumerThread(es_client_thread, config.index, q) for _ in range(N_THREADS)]

    for i in range(N_THREADS):
        threads[i].start()
