#!/usr/bin/env python
# coding: utf-8
import sys; sys.path.append("/home/vin/Projects/nir/")
import os
import dill

from functools import partial
from training.utils import DatasetToSentTrans
from training.hparm_tuning.optuna_rank import run_optuna_with_wandb, print_optuna_stats
from training.hparm_tuning.trainer import SentenceTransformerHparamTrainer

from sentence_transformers import evaluation
from datasets.trec_clinical_trials import TrecClinicalTrialTripletParser
from core import ParsedTopicsToDataset
from training.hparm_tuning.config import HparamConfig


TASK_NAME = "trec_contrastive_passage"
OUTPUT_DIR = f"./outputs/cross_encoder/{TASK_NAME}/"
DATA_DIR = "../data/"


def get_dataset(dataset_save_path="."):
    os.makedirs(dataset_save_path, exist_ok=True)

    train_fp = os.path.join(dataset_save_path, "train.dill")
    val_fp = os.path.join(dataset_save_path, "val.dill")

    if not os.path.isfile(train_fp) and not os.path.isfile(val_fp):
        topics = TrecClinicalTrialTripletParser.get_topics(
            "/home/vin/Datasets/clinical_trials/topics/enriched_trec.json")
        converted_dataset = ParsedTopicsToDataset.convert(TrecClinicalTrialTripletParser, topics)

        converted_dataset = converted_dataset.map(lambda example: {"label": int(int(example['rel']) > 0)})
        # converted_dataset = converted_dataset.shuffle().train_test_split(test_size=0.25)
        converted_dataset = converted_dataset.shuffle().train_test_split(test_size=0.4)
        converted_dataset = converted_dataset['test']
        converted_dataset = converted_dataset.shuffle().train_test_split(test_size=0.2)

        train_dataset = DatasetToSentTrans(converted_dataset['train'], text_cols=["q_text", "brief_title"],
                                           label_col="label")
        val_dataset = DatasetToSentTrans(converted_dataset['test'], text_cols=["q_text", "brief_title"],
                                         label_col="label")

        dill.dump(train_dataset, open(train_fp, "wb+"))
        dill.dump(val_dataset, open(val_fp, "wb+"))
    else:
        train_dataset = dill.load(open(train_fp, "rb"))
        val_dataset = dill.load(open(val_fp, "rb"))

    return {"train": train_dataset, "val": val_dataset}


if __name__ == "__main__":
    hparam_config = HparamConfig.from_json(
        "./configs/hparam/trec2021_tuning.json"
    )

    trainer = SentenceTransformerHparamTrainer(
        dataset_loading_fn=partial(get_dataset, "."),
        evaluator_fn=evaluation.BinaryClassificationEvaluator,
        hparams_config=hparam_config,
    )

    study = run_optuna_with_wandb(trainer, wandb_kwargs={"project": "trec2021-tuning-new"})
    print_optuna_stats(study)
