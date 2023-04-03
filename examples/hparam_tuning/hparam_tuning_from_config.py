# coding: utf-8

import datasets
import plac
from sentence_transformers import evaluation

from debeir.training.hparm_tuning.config import HparamConfig
from debeir.training.hparm_tuning.optuna_rank import print_optuna_stats, run_optuna_with_wandb
from debeir.training.hparm_tuning.trainer import SentenceTransformerHparamTrainer
from debeir.training.utils import SentDataset
from debeir.utils.lazy_static import lazy_static

TASK_NAME = "trec_contrastive_passage"
OUTPUT_DIR = f"./outputs/cross_encoder/{TASK_NAME}/"
DATA_DIR = "../data/"


def remap_label(ex):
    # Normalize 0, 1, 2 -> 0, 1
    # We treat neutral the same as a contradiction
    ex['label'] = ex['label'] // 2

    return ex


def load_dataset(limit):
    """
    Load and preprocess the SNLI dataset

    1. Re-normalize the labels to binary
    2. Remove examples with no gold labels.
    """

    dataset = datasets.load_dataset('snli')

    # Use our sentence transformer dataset adapter pattern, allows for huggingface datasets to be used with
    # Sentence transformer API
    select_range = limit if limit != -1 else len(dataset['train'])

    train = SentDataset(
        dataset['train'].select(range(select_range)).map(remap_label).filter(lambda k: k['label'] != -1),
        text_cols=['premise', 'hypothesis'],
        label_col='label')

    select_range = limit if limit != -1 else len(dataset['test'])

    val = SentDataset(dataset['test'].select(range(select_range)).map(remap_label).filter(lambda k: k['label'] != -1),
                      text_cols=['premise', 'hypothesis'],
                      label_col='label')

    # Make sure our validation and train column name are correct for the trainer
    return {'train': train, 'val': val}


@plac.opt('limit', "Only sample a subset for training", type=int)
def main(limit=-1):
    if limit != -1:
        print(f"Limiting training and validation examples to {limit}")
    else:
        print(f"Running full test sample of snli, using the flag --limit [number] to limit the training instances.")

    hparam_config = HparamConfig.from_json(
        "hparam_cfg.json"
    )

    trainer = SentenceTransformerHparamTrainer(
        # Reuse the dataset with lazy static, so we don't have to do the preprocessing repeatedly
        dataset_loading_fn=lazy_static("hparam_dataloader", load_dataset, limit),
        evaluator_fn=evaluation.BinaryClassificationEvaluator,
        hparams_config=hparam_config,
    )

    study = run_optuna_with_wandb(trainer)
    print_optuna_stats(study)


if __name__ == "__main__":
    plac.call(main)
