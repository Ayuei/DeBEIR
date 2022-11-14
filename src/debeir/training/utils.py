from typing import List, Union

import datasets
import loguru
from sentence_transformers import SentenceTransformer, losses, models
from sentence_transformers.evaluation import SentenceEvaluator, EmbeddingSimilarityEvaluator
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from wandb import wandb
import transformers

from debeir.data_sets.types import RelevanceExample, InputExample
# from sentence_transformers import InputExample
from datasets import concatenate_datasets


class LoggingScheduler:
    def __init__(self, scheduler: LambdaLR):
        self.scheduler = scheduler

    def step(self, epoch=None):
        self.scheduler.step(epoch)

        last_lr = self.scheduler.get_last_lr()

        for i, lr in enumerate(last_lr):
            wandb.log({f"lr_{i}": lr})

    def __getattr__(self, attr):
        return getattr(self.scheduler, attr)


def get_scheduler_with_wandb(optimizer, scheduler: str, warmup_steps: int, t_total: int):
    """
    Returns the correct learning rate scheduler. Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
    """
    scheduler = scheduler.lower()
    loguru.logger.info(f"Creating scheduler: {scheduler}")

    if scheduler == 'constantlr':
        sched = transformers.get_constant_schedule(optimizer)
    elif scheduler == 'warmupconstant':
        sched = transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    elif scheduler == 'warmuplinear':
        sched = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                             num_training_steps=t_total)
    elif scheduler == 'warmupcosine':
        sched = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                             num_training_steps=t_total)
    elif scheduler == 'warmupcosinewithhardrestarts':
        sched = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                                num_warmup_steps=warmup_steps,
                                                                                num_training_steps=t_total)
    else:
        raise ValueError("Unknown scheduler {}".format(scheduler))

    return LoggingScheduler(sched)


class LoggingLoss:
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def __call__(self, *args, **kwargs):
        loss = self.loss_fn(*args, **kwargs)
        wandb.log({'train_loss': loss})
        return loss

    def __getattr__(self, attr):
        return getattr(self.loss_fn, attr)


class TokenizerOverload:
    def __init__(self, tokenizer, tokenizer_kwargs, debug=False):
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
        self.debug = debug
        self.max_length = -1

    def __call__(self, *args, **kwargs):
        if self.debug:
            print(str(args), str(kwargs))

        kwargs.update(self.tokenizer_kwargs)
        output = self.tokenizer(*args, **kwargs)

        return output

    def __getattr__(self, attr):
        if self.debug:
            print(str(attr))

        return getattr(self.tokenizer, attr)


class LoggingEvaluator:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def __call__(self, *args, **kwargs):
        scores = self.evaluator(*args, **kwargs)
        wandb.log({'val_acc': scores})

        return scores

    def __getattr__(self, attr):
        return getattr(self.evaluator, attr)


class SentDataset:
    def __init__(self, dataset: datasets.Dataset, text_cols: List[str],
                 label_col: str = None, label=None):
        self.dataset = dataset
        self.text_cols = text_cols
        self.label_col = label_col
        self.label = label

    def __getitem__(self, idx):
        item = self.dataset[idx]

        texts = []

        for text_col in self.text_cols:
            texts.append(item[text_col])

        example = InputExample(texts=texts)

        if self.label_col:
            example.label = item[self.label_col]
        else:
            if self.label:
                example.label = self.label

        return example

    def __len__(self):
        return len(self.dataset)


class SentDatasetList:
    def __init__(self, datasets: List[SentDataset]):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in self.datasets]
        self.total_length = sum(self.lengths)

    def __getitem__(self, idx):
        i, c = 0, 0

        for i, length in enumerate(self.lengths):
            if idx - c == 0:
                idx = 0
                break
            if idx - c < length:
                idx = idx - c
                break

            c = c + self.lengths[i]

        return self.datasets[i][idx]

    def __len__(self):
        return self.total_length


def _train_sentence_transformer(model_fp_or_name: str, output_dir: str,
                                train_dataset: List[Union[RelevanceExample, InputExample]],
                                eval_dataset: List[Union[RelevanceExample, InputExample]],
                                train_batch_size=32, num_epochs=3,
                                warmup_steps=None, evaluate_every_n_step: int = 1000, special_tokens=None,
                                pooling_mode=None, loss_func=None, evaluator: SentenceEvaluator = None):
    """
        Train a sentence transformer model

        Returns the model for evaluation
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

    return model


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


def get_max_seq_length(tokenizer, dataset, x_labels, dataset_key="train"):
    dataset = dataset.map(lambda example: tokenizer([example[x_label] for x_label in x_labels]))

    max_length = -1
    for example in dataset[dataset_key]['attention_mask']:
        length = max(sum(x) for x in example)
        if length > max_length:
            max_length = length

    return max_length
