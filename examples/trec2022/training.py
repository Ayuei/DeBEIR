#!/usr/bin/env python
# coding: utf-8
import sys;

import wandb

sys.path.append("/home/vin/Projects/nir/")
import os
import dill

from training.utils import DatasetToSentTrans
from training.hparm_tuning.trainer import SentenceTransformerTrainer

from sentence_transformers import evaluation
from data_sets.trec_clinical_trials import TrecClinicalTrialTripletParser
from interfaces import ParsedTopicsToDataset
from training.hparm_tuning.config import HparamConfig


def get_dataset(dataset_save_path=".", override=False):
    os.makedirs(dataset_save_path, exist_ok=True)

    train_fp = os.path.join(dataset_save_path, "train.dill")
    val_fp = os.path.join(dataset_save_path, "val.dill")

    if override or (not os.path.isfile(train_fp) and not os.path.isfile(val_fp)):
        topics = TrecClinicalTrialTripletParser.get_topics(
            "/home/vin/Datasets/clinical_trials/topics/enriched_trec.json")
        converted_dataset = ParsedTopicsToDataset.convert(TrecClinicalTrialTripletParser, topics)

        converted_dataset = converted_dataset.map(lambda example: {"label": int(int(example['rel']) > 0)})
        # Map the topic ids, and then filter based on topic id
        converted_dataset = converted_dataset.shuffle().train_test_split(test_size=0.15)
        dill.dump(converted_dataset, open("converted_dataset.dill", "wb+"))

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
        "./configs/training/submission.json"
    )

    wandb.wandb.init(project="trec2022_submission")

    trainer = SentenceTransformerTrainer(
        dataset=get_dataset(override=True),
        evaluator_fn=evaluation.BinaryClassificationEvaluator,
        hparams_config=hparam_config,
        use_wandb=True
    )

    trainer.fit(
        save_best_model=True,
        checkpoint_save_steps=179
    )
