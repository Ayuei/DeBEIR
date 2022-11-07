"""
The DeBEIR (Dense Bi-Encoder Information Retrieval) source code library.

See ./main.py in the parent directory for an out-of-the-box runnable code.

Otherwise, check out notebooks in the parent directory for training your own model amongst other things.
"""

from .interfaces.pipeline import Pipeline, NIRPipeline
from .interfaces.document import Document
from .interfaces.query import Query
from .interfaces.config import Config