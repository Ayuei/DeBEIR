import asyncio
import pytest

import main


def test_queries():
    asyncio.run(
        main.main(
            topics="../assets/topics-rnd5.xml",
            configs=["../configs/trec_covid/embedding.toml"],
            nir_config="../configs/nir.toml",
            elasticsearch=True)
    )

    # async def main(
    #        topics,
    #        configs,
    #        debug=False,
    #        **kwargs
    # )
    #    parser.add_argument('--topics', help="Path to topic file")
    #    parser.add_argument('--configs', nargs='*',
    #                        help="Run Config to run, can add multiple configs")
    #    parser.add_argument('--nir_config', default="./configs/nir.toml",
    #                        help="NIR Server Config")
    #    parser.add_argument('--elasticsearch', action='store_true', help="Use the Elasticsearch Engine")
    #    parser.add_argument('--solr', action='store_true', help="Use the Lucene Solr Engine")
    #    parser.add_argument('--debug', action='store_true', help="Turn on debug logging"
