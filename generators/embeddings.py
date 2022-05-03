import sentence_transformers
import torch
import torch.nn.functional as F
import spacy
import scispacy
from cache import Cache


class Encoder:
    def __init__(self, model_path, normalize=True, spacy_model="en_core_sci_sm", max_length=2000000):
        self.model = sentence_transformers.SentenceTransformer(model_path)
        self.model_path = model_path
        self.nlp = spacy.load(spacy_model)
        self.spacy_model = spacy_model
        self.max_length = max_length
        self.nlp.max_length = max_length
        self.normalize = normalize

    @Cache
    def encode(self, topic):
        sentences = [' '.join(sent.text.split()) for sent in self.nlp(topic).sents if sent.text.strip()]
        embeddings = self.model.encode(sentences, convert_to_tensor=True)

        if len(embeddings.size()) == 1:
            embeddings = torch.unsqueeze(embeddings, dim=0)
            embeddings = torch.mean(embeddings, axis=0)

        if self.normalize:
            embeddings = F.normalize(embeddings, dim=0)

        embeddings = embeddings.tolist()

        if isinstance(embeddings, list) and isinstance(embeddings[0], list):
            return embeddings[0]

        return embeddings

    def __hash__(self):
        return hash((self.model_path, self.spacy_model, self.normalize, self.max_length))
