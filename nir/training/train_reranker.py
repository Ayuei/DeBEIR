from typing import List

from sentence_transformers.evaluation import SentenceEvaluator
from nir.training.utils import _train_sentence_transformer
from nir.data_sets.types import RelevanceExample


def train_cross_encoder_reranker(model_fp_or_name: str, output_dir: str, train_dataset: List[RelevanceExample],
                                 dev_dataset: List[RelevanceExample], train_batch_size=32, num_epochs=3, warmup_steps=None,
                                 evaluate_every_n_step: int = 1000,
                                 special_tokens=None, pooling_mode=None, loss_func=None,
                                 evaluator: SentenceEvaluator = None,
                                 *args, **kwargs):
    """
    Trains a reranker with relevance signals

    :param model_fp_or_name: The model name or path to the model
    :param output_dir: Output directory to save model, logs etc.
    :param train_dataset: Training Examples
    :param dev_dataset: Dev examples
    :param train_batch_size: Training batch size
    :param num_epochs: Number of epochs
    :param warmup_steps: Warmup steps for the scheduler
    :param evaluate_every_n_step: Evaluate the model every n steps
    :param special_tokens: Special tokens to add, defaults to [DOC], [QRY] tokens (bi-encoder)
    :param pooling_mode: Pooling mode for a sentence transformer model
    :param loss_func: Loss function(s) to use
    :param evaluator: Evaluator to use
    """

    if special_tokens is None:
        special_tokens = ["[DOC]", "[QRY]"]

    return _train_sentence_transformer(model_fp_or_name, output_dir, train_dataset,
                                       dev_dataset, train_batch_size,
                                       num_epochs, warmup_steps, evaluate_every_n_step,
                                       special_tokens, pooling_mode, loss_func,
                                       evaluator)
