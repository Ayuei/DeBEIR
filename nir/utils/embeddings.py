import sentence_transformers
import torch
import torch.nn.functional as F
import spacy
from analysis_tools_ir.utils import cache
from hashlib import md5

EMBEDDING_DIM_SIZE = 768


class Encoder:
    def __init__(
            self,
            model_path,
            normalize=False,
            spacy_model="en_core_sci_md",
            max_length=2000000,
    ):
        self.model = sentence_transformers.SentenceTransformer(model_path)
        self.model_path = model_path
        self.nlp = spacy.load(spacy_model)
        self.spacy_model = spacy_model
        self.max_length = max_length
        self.nlp.max_length = max_length
        self.normalize = normalize

    @cache.Cache(hash_self=True, cache_dir="./cache/embedding_cache/")
    def encode(self, topic):
        sentences = [
            " ".join(sent.text.split())
            for sent in self.nlp(topic).sents
            if sent.text.strip()
        ]
        embeddings = self.model.encode(sentences, convert_to_tensor=True,
                                       show_progress_bar=False)

        if len(embeddings.size()) == 1:
            embeddings = torch.unsqueeze(embeddings, dim=0)
            embeddings = torch.mean(embeddings, axis=0)

        if self.normalize:
            embeddings = F.normalize(embeddings, dim=-1)

        embeddings = embeddings.tolist()

        if isinstance(embeddings, list) and isinstance(embeddings[0], list):
            return embeddings[0]

        return embeddings

    def __eq__(self, other):
        return (
                self.model_path == other.model_path
                and self.spacy_model == other.spacy_model
                and self.normalize == other.normalize
                and self.max_length == other.max_length
        )

    def __hash__(self):
        return int(
            md5(
                (self.model_path
                 + self.spacy_model
                 + str(self.normalize)
                 + str(self.max_length)
                 ).encode()
            ).hexdigest(),
            16,
        )
