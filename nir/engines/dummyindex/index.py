import elasticsearch


class DummyIndex:
    def get_documents(self):
        pass

    def query(self):
        pass

    def scorer(self):
        pass


async def es_isup(es_client: elasticsearch.AsyncElasticsearch):
    return await es_client.ping()
