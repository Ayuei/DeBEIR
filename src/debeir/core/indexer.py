import abc
import threading
from queue import Queue
from typing import List

from debeir.rankers.transformer_sent_encoder import Encoder
from debeir.utils.utils import remove_excess_whitespace
from elasticsearch import Elasticsearch


class Indexer:
    def __init__(self, client):
        super().__init__()
        self.client = client

    @abc.abstractmethod
    def get_field(self, document, field):
        pass


class SemanticElasticsearchIndexer(Indexer, threading.Thread):
    """
    Create a NIR-style index, with dense field representations with provided sentence encoder
    Assumes you've already indexed to start with.
    """

    def __init__(self, es_client: Elasticsearch, encoder: Encoder, index: str,
                 fields_to_encode: List[str], queue: Queue):
        super().__init__(es_client)
        self.encoder = encoder
        self.index = index
        self.fields = fields_to_encode
        self.q = queue
        self.update_mappings(self.index, self.fields, self.client)

    @classmethod
    def update_mappings(self, index, fields, client: Elasticsearch):
        mapping = {}
        value = {
            "type": "dense_vector",
            "dims": 768
        }

        for field in fields:
            mapping[field + "_Embedding"] = value
            mapping[field + "_Text"] = {"type": "text"}

        client.indices.put_mapping(
            body={
                "properties": mapping
            }, index=index)

    # async def create_index(self, document_itr=None):
    #    await self._update_mappings()

    #    if document_itr is None:
    #        document_itr = helpers.async_scan(self.es_client, index=self.index)

    #    bar = tqdm(desc="Indexing", total=35_000)

    #    async for document in document_itr:
    #        doc = document["_source"]
    #        await self.index_document(doc)

    #        bar.update(1)

    def get_field(self, document, field):
        if field not in document:
            return False

        if "f{field}_Text" in document and document["f{field}_Text"] != 0:
            return False

        if 'Textblock' in document[field]:
            return remove_excess_whitespace(document[field]['Textblock'])

        return remove_excess_whitespace(document[field])

    def index_document(self, document):
        update_doc = {}
        doc = document["_source"]

        for field in self.fields:
            text_field = self.get_field(doc, field)

            if text_field:
                embedding = self.encoder.encode(topic=text_field, disable_cache=True)
                update_doc[f"{field}_Embedding"] = embedding
                update_doc[f"{field}_Text"] = text_field

        if update_doc:
            self.client.update(index=self.index,
                               id=document['_id'],
                               doc=update_doc)

    def run(self):
        while not self.q.empty():
            document = self.q.get()
            self.index_document(document)
