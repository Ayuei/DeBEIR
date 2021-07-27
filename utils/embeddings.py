import sentence_transformers
import asyncio
import torch
import torch.nn.functional as F


class Encoder:
    def __init__(self, model_path):
        self.model = sentence_transformers.SentenceTransformer(model_path)

    async def encode(self, sentences, normalize=True):
        await asyncio.sleep(0.01)

        embeddings = self.model.encode(sentences, convert_to_tensor=True)

        if len(embeddings.size()) == 1:
            embeddings = torch.unsqueeze(embeddings, dim=0)
            embeddings = torch.mean(embeddings, axis=0)

        if normalize:
            norm = F.normalize(embeddings, dim=0)

            return norm.tolist()

        return embeddings.tolist()
