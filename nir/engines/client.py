import dataclasses
from elasticsearch import AsyncElasticsearch


@dataclasses.dataclass(init=True)
class Client:
    """
    Overarching client interface object that contains references to different clients for search
    Allows sharing between function calls
    """
    es_client: AsyncElasticsearch = None
    solr_client: object = None
    generic_client: object = None

    async def close(self):
        """
        Generically close all contained client objects
        """
        if self.es_client:
            await self.es_client.close()

        if self.solr_client:
            await self.solr_client.close()

        if self.generic_client:
            await self.generic_client.close()
