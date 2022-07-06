from functools import partial
from typing import List, Union

import nir_datasets
import transformers

from sentence_transformers.evaluation import SentenceEvaluator
from transformers import SchedulerType, Trainer, AutoModel, TrainingArguments

from rankers.training.utils import _train_sentence_transformer, tokenize_function
from nir_datasets.utils import InputExample, RelevanceExample


def train_biencoder(model_fp_or_name: str, output_dir: str, train_examples: List[InputExample],
                    dev_examples: List[InputExample], train_batch_size=32, num_epochs=3, warmup_steps=None,
                    evaluate_every_n_step: int = 1000,
                    special_tokens=None, pooling_mode=None, loss_func=None,
                    evaluator: SentenceEvaluator = None, *args, **kwargs):
    """
    Train a universal sentence encoder

    :param model_fp_or_name: The model name or path to the model
    :param output_dir: Output directory to save model, logs etc.
    :param train_examples: Training Examples
    :param dev_examples: Dev examples
    :param train_batch_size: Training batch size
    :param num_epochs: Number of epochs
    :param warmup_steps: Warmup steps for the scheduler
    :param evaluate_every_n_step: Evaluate the model every n steps
    :param special_tokens: Special tokens to add
    :param pooling_mode: Pooling mode for a sentence transformer model
    :param loss_func: Loss function(s) to use
    :param evaluator: Evaluator to use
    """

    return _train_sentence_transformer(model_fp_or_name, output_dir, train_examples, dev_examples, train_batch_size,
                                       num_epochs, warmup_steps, evaluate_every_n_step, special_tokens,
                                       pooling_mode, loss_func, evaluator)


def train_huggingface_transformer(model_fp_or_name_or_cls: Union[str, transformers.PreTrainedModel],
                                  tokenizer: transformers.PreTrainedTokenizer,
                                  output_dir: str,
                                  compute_metric_fn,
                                  metric: nir_datasets.Metric,
                                  dataset: nir_datasets.DatasetDict = None,
                                  train_dataset: List[Union[RelevanceExample, InputExample, nir_datasets.Dataset]] = None,
                                  eval_dataset: List[Union[RelevanceExample, InputExample, nir_datasets.Dataset]] = None,
                                  train_batch_size=32, num_epochs=3,
                                  learning_rate=5e-5,
                                  lr_scheduler_type: SchedulerType = SchedulerType.CONSTANT_WITH_WARMUP,
                                  optimizer: str = "adamw_hf",
                                  warmup_ratio=0.1, evaluate_every_n_step: int = 1000,
                                  pooling_mode=None, loss_func=None,
                                  model_args=None, model_kwargs=None,
                                  padding_strategy="max_length",
                                  truncate=True,
                                  special_tokens=None,
                                  seed=42,
                                  *args, **kwargs) -> Trainer:
    """
    Train a transformer model using the Huggingface API

    :param model_fp_or_name_or_cls: Model name or model class to instantiate
    :param tokenizer: Tokenizer
    :param output_dir: Output directory to write to
    :param compute_metric_fn: Metric function to compute metrics
    :param metric: Metric used by the compute_metric_fn
    :param dataset: Huggingface Dataset Dict
    :param train_dataset: Training dataset to be used by the Trainer class
    :param eval_dataset: Evaluation dataset to be used by the Trainer class
    :param train_batch_size: Batch size to use for training
    :param num_epochs: Number of training epochs (default: 3)
    :param learning_rate: Learning rate (default: 5e-5)
    :param lr_scheduler_type: Learning rate type, see SchedulerType
    :param optimizer: Optimizer
    :param warmup_ratio: Warmup ratios as ratio of steps (default 0.1)
    :param evaluate_every_n_step: Number of steps to evaluate
    :param pooling_mode: Pooling mode for your model
    :param loss_func: Loss function to instantiate model
    :param model_args: Model arguments to pass
    :param model_kwargs: Model keyword arguments
    :param padding_strategy: Tokenization padding strategy
    :param truncate: Truncate tokenization strategy
    :param special_tokens: Special tokens to add to the tokenizer
    :param seed: Dataset shuffle seed
    :param args:
    :param kwargs:
    :return:
    """

    if isinstance(model_fp_or_name_or_cls, str):
        model = AutoModel.from_pretrained(model_fp_or_name_or_cls)
    elif isinstance(model_fp_or_name_or_cls, type):
        # is already instantiated
        model = model_fp_or_name_or_cls
    else:
        # Is not instantiated
        model = model_fp_or_name_or_cls(loss_func=loss_func,
                                        pooling_mode=pooling_mode,
                                        *model_args, **model_kwargs)

    if special_tokens:
        tokenizer.add_tokens(special_tokens, special_tokens=True)
        tokenizer.resize_token_embeddings(len(tokenizer))

    tokenized_datasets = dataset.map(
        partial(
            tokenize_function, tokenizer,
            padding_strategy=padding_strategy,
            truncate=truncate
        ), batched=True)

    if dataset:
        train_dataset = tokenized_datasets["train"].shuffle(seed=seed)
        eval_dataset = tokenized_datasets["dev"].shuffle(seed=seed)

    training_args = TrainingArguments(output_dir=output_dir,
                                      per_gpu_train_batch_size=train_batch_size,
                                      num_train_epochs=num_epochs,
                                      warmup_ratio=warmup_ratio,
                                      eval_steps=evaluate_every_n_step,
                                      learning_rate=learning_rate,
                                      lr_scheduler_type=lr_scheduler_type,
                                      optim=optimizer,
                                      fp16=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=partial(compute_metric_fn, metric),
    )

    trainer.fit()

    return trainer
