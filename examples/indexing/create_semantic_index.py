import asyncio
from typing import List

from elasticsearch import AsyncElasticsearch, helpers

from data_sets.factory import apply_nir_config
from interfaces.config import GenericConfig
from nir.rankers.transformer_sent_encoder import Encoder


class SemanticIndexer:
    """
    Create a NIR-style index, with dense field representations with provided sentence encoder
    """

    def __init__(self, es_client: AsyncElasticsearch, encoder: Encoder,
                 index: str, fields_to_encode: List[str]):
        self.es_client = es_client
        self.encoder = encoder
        self.index = index
        self.fields = fields_to_encode

    async def _update_mappings(self):
        mapping = {}
        value = {
            "type": "dense_vector",
            "dims": 768
        }

        for field in self.fields:
            mapping[field + "_Embedding"] = value

        await self.es_client.indices.put_mapping(
            body={
                "properties": mapping
            }, index=self.index)

    async def create_index(self, document_itr=None):
        await self._update_mappings()

        if document_itr is None:
            document_itr = helpers.async_scan(self.es_client, index=self.index)

        async for document in document_itr:
            update_doc = {}

            for field in self.fields:
                embedding = self.encoder.encode(document[field])
                update_doc[f"{field}_Embedding"] = embedding

            await self.es_client.update(index=self.index,
                                        id=document['_id'],
                                        doc=update_doc)


@apply_nir_config
def get_config_kwargs(nir_config, **kwargs):
    return kwargs


if __name__ == "__main__":
    kwargs = get_config_kwargs(nir_config="./configs/nir.toml")

    config = GenericConfig.from_toml(
        fp="./configs/trec_covid/embedding.toml",
        field_class=GenericConfig
    )

    es_client = AsyncElasticsearch(f"{kwargs['protocol']}://{kwargs['ip']}:{kwargs['port']}",
                                   timeout=kwargs['timeout'])

    encoder = Encoder(model_path="/home/vin/Projects/nir/outputs/submission/trec_model")
    indexer = SemanticIndexer(es_client,
                              encoder=config.encoder,
                              index=config.index,
                              fields_to_encode=['brief_title']
                              )
