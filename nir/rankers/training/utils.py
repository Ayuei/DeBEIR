from typing import List, Union

from sentence_transformers import SentenceTransformer, losses, models
from sentence_transformers.evaluation import SentenceEvaluator, EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

from nir_datasets.utils import InputExample, RelevanceExample


def _train_sentence_transformer(model_fp_or_name: str, output_dir: str,
                                train_dataset: List[Union[RelevanceExample, InputExample]],
                                eval_dataset: List[Union[RelevanceExample, InputExample]],
                                train_batch_size=32, num_epochs=3,
                                warmup_steps=None, evaluate_every_n_step: int = 1000, special_tokens=None,
                                pooling_mode=None, loss_func=None, evaluator: SentenceEvaluator = None):
    """
        Train a sentence transformer model
    """

    encoder = models.Transformer(model_fp_or_name)

    if special_tokens:
        encoder.tokenizer.add_tokens(special_tokens, special_tokens=True)
        encoder.auto_model.resize_token_embeddings(len(encoder.tokenizer))

    pooling_model = models.Pooling(encoder.get_word_embedding_dimension(),
                                   pooling_mode=pooling_mode)

    model = SentenceTransformer(modules=[encoder, pooling_model])

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)

    if loss_func is None:
        loss_func = losses.CosineSimilarityLoss(model=model)

    if evaluator is None:
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(eval_dataset)

    model.fit(train_objectives=[(train_dataloader, loss_func)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=evaluate_every_n_step,
              warmup_steps=warmup_steps if warmup_steps else (num_epochs * len(train_dataset)) // 20,
              output_path=output_dir)


def tokenize_function(tokenizer, examples, padding_strategy, truncate):
    """
    Tokenizer function

    :param tokenizer: Tokenizer
    :param examples: Input examples to tokenize
    :param padding_strategy: Padding strategy
    :param truncate: Truncate sentences
    :return:
        Returns a list of tokenized examples
    """
    return tokenizer(examples["text"],
                     padding=padding_strategy,
                     truncation=truncate)
