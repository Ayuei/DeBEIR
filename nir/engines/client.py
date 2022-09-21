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

    @classmethod
    def build_from_config(cls, engine_type, engine_config) -> 'Client':
        """
        Build client from engine config
        :param engine_type:
        :param engine_config:
        :return:
        """

        client = Client()

        if engine_type == "elasticsearch":
            es_client = AsyncElasticsearch(
                f"{engine_config['protocol']}://{engine_config.ip}:{engine_config.port}",
                timeout=engine_config.timeout
            )

            client.es_client = es_client

        return client

    def get_client(self, engine):
        if engine == "elasticsearch":
            return self.es_client

    def close(self):
        """
        Generically close all contained client objects
        """
        if self.es_client:
            await self.es_client.close()

        if self.solr_client:
            await self.solr_client.close()

        if self.generic_client:
            await self.generic_client.close()
