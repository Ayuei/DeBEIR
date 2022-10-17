---
title: 'DeBIR: A Python Package for Dense Bi-Encoder Information Retrieval'
tags:
  - information retrieval
  - dense retrieval
  - bi-encoder 
  - transformers
  - pytorch
  - python
  - deep learning
  - neural networks
  - machine learning
  - natural language processing 
authors:
  - name: Vincent Nguyen
    orcid: 0000-0003-1787-8090 
    affiliation: "1, 2"
  - name: Sarvnaz Karimi 
    orcid: 0000-0002-4927-3937 
    affiliation: "2"
  - name: Zhenchang Xing 
    orcid: 0000-0001-7663-1421 
    affiliation: "1, 2"
affiliations:
 - name: Australian National University 
   index: 1
 - name: "CSIRO's Data61"
   index: 2
date: 4 October 2022 
bibliography: paper.bib
---

# Summary
Information Retrieval (IR) is the task of retrieving documents given a query or information need. These documents are retrieved and ranked based on a relevance function or relevance model such as Best-Matching 25 [@bm25]. Although deep learning has been successful in other computer science fields such as computer vision [@alexnet; @googlenet] and natural language processing [@orig-bert-2018; @liu:2019; @biobert], success in information retrieval was limited until 2021 [@lin-neural-recantation]. In 2021, deep learning in information retrieval could surpass less computationally intensive keyword-based statistical models in terms of retrieval effectiveness [@yang2019critically], sparking the field of dense retrieval. Dense retrieval is the task of retrieving documents given a query or information need using a dense vector representation of the query and documents. The dense vector representation is obtained by passing the query and documents through a neural network. The neural network is usually a pre-trained language model such as BERT [@orig-bert-2018] or RoBERTa [@roberta]. The dense query vector representation is then used to retrieve documents using a similarity function such as cosine similarity.

Unlike statistical learning, tuning of deep learning retrieval methods is often costly and time-consuming. This makes it important to efficiently automate much of the training, tuning and evaluation processes. 

DeBIR is a library for facilitating dense retrieval research, with a primary focus on bi-encoder dense retrieval where query and documents dense vectors are generated separately [@reimers2019]. It allows for expedited experimentation in dense retrieval research by reducing boilerplate code through an interchangeable pipeline API and code extendability through inheritance of general classes. It further abstracts common training loops and hyperparameter tuning into easy-to-define configuration files. This library is aimed at helping practitioners, researchers and data scientists experimenting with providing them dense retrieval methods that are easy-to-use out of the box but also have additional hackability for more nuanced research.

A brief summary of the pipeline stages are:

1. Configuration based on TOML files, these are loaded in a class factory to create pipeline objects.

2. An executor object takes in a query builder object. The purpose of the query builder object is to define the mapping of the documents and which parts of the query to use for query execution.

3. The executor object asynchronously executes the queries.

4. Finally, an evaluator object uses the results for a listing of metrics defined by a configuration file against an oracle test set.


This pipeline is condensed into a single class that can be built from a configuration file.

# Statement of Need
Dense retrieval has been popular in Information Retrieval for some time [@drmm; @abcnn; @paccr]. In the early 2000s, there had been considerable stagnation in retrieval effectiveness as there was a lack of strong baselines [@Armstrong:2009] when comparing new methods. This observation was repeated with the rise of deep learning, where retrieval performance was again compared against weaker baselines and were not significantly stronger than older statistical models, such as a well-tuned BM25 model [@yang2019critically].

However, this was later recanted when transformer models could be used fine-tuned on Natural Language Inference tasks or ms-marco as a cross-encoder (where a query and document pair are encoded at ranking time) [@lin-neural-recantation], significantly overtaking even the best BM25 models. 

Cross-encoder models have demonstrated to be highly succesful. However, the use of bi-encoders is still under-used in the IR field. This is most likely due to the fact that cross-encoders often obtain higher retrieval effectiveness. However, the downside to cross-encoders is the need to encode the query and document pair at ranking time. This is computationally expensive and can be a bottleneck in production systems. Bi-encoders, on the other hand, can be used to encode the query and document separately, and then use a similarity function to retrieve documents. This is computationally cheaper and can be used in production systems. Furthermore, by combining with logarithmically with BM25 models, cross encoders have demonstrated strong retrieval effectiveness in biomedical research retrieval [@search-like-an-expert-2022].

DeBIR is a library that mainly facilitates bi-encoder research (where query and document can be encoded independently) and provides base classes with flexible functionality through inheritance. Although we provide re-rankers for cross-encoders, the priority of the library is to facilitate bi-encoder research. Where the strength of bi-encoders lies in the use of offline indexing of dense vectors.

The DeBIR library exposes an API for commonly used functions for training, hyper-parameter tuning (Figure \autoref{fig:training} and evaluation of transformer-based models. The pipeline can be broken up into multiple stages: parsing, query building, query execution, serialization and evaluation (Figure \autoref{fig:pipeline}. Furthermore, we package a caching mechanism for the expensive encoding operations, which can be used to speed up the pipeline during repeated experimentation. 

We believe this tool will be useful for facilitating research and rapid experimentation and will continue to improve the library.

Similar libraries that exist include sentence-transformers, openNIR, but have less of a focus on all stages of the dense retrieval pipeline, limited extendability (we provide base classes that can be inherited) or are tailored to general purpose machine learning.

![Standard flow of the DeBIR query/evaluation loop.\label{fig:pipeline}](pipeline.pdf){scale=0.5}

![Standard flow of the DeBIR training loop.\label{fig:training}](training.pdf){scale=0.43}

# Acknowledgments
The DeBIR library uses sentence-transformers, hugggingface's transformers and datasets, allRank, optuna, elasticsearch and trectools python packages.

This search is supported by CSIRO Data61, an Australian Government agency through the Precision Medicine FSP program and the Australian Research Training Program. We extend thanks to Brian Jin (Data61) for providing a code review.

# Examples
### Pipeline
The pipeline is a single class that can be built from a configuration file. The configuration file is a TOML file that defines the pipeline stages and their parameters. The pipeline is built using a class factory that takes in the configuration file and creates the pipeline stages. The pipeline stages are then executed in order.

```python
from debeir.interfaces.pipeline import NIRPipeline
from debir.interfaces.callbacks import (SerializationCallback, 
                                        EvaluationCallback)
from debir.evaluation import Evaluator

p = NIRPipeline.build_from_config(config_fp="./tests/config.toml", 
                                  engine="elasticsearch",
                                  nir_config_fp="./tests/nir_config.toml")

# Optional callbacks to serialize to disk
serial_cb = SerializationCallback(p.config, p.nir_settings)

# Or evaluation
evaluator = Evaluator.build_from_config(p.config, metrics_config=p.metrics_config)
evaluate_cb = EvaluationCallback(evaluator, 
                                 config=p.config)

p.add_callback(serial_cb)
p.add_callback(evaluate_cb)

# Asynchronously execute queries
results = await p.run_pipeline()

# Post processing of results can go here
```

### Training a model

```python
import wandb

from debeir.training.hparm_tuning.trainer import SentenceTransformerTrainer
from debeir.training.hparm_tuning.config import HparamConfig
from sentence_transformers import evaluation

# Load a hyper-parameter configuration file
hparam_config = HparamConfig.from_json(
        "./configs/training/submission.json"
)

# Integration with wandb 
wandb.wandb.init(project="My Project")

# Create a trainer object
trainer = SentenceTransformerTrainer(
    dataset=get_dataset(), # Specify some dataloading function here
    evaluator_fn=evaluation.BinaryClassificationEvaluator,
    hparams_config=hparam_config,
    use_wandb=True
)

# Foward parameters to underlying SentenceTransformer model
trainer.fit(
    save_best_model=True,
    checkpoint_save_steps=179
)
```

### Hyperparameter tuning

```python
from sentence_transformers import evaluation
from debeir.training.hparm_tuning.optuna_rank import (run_optuna_with_wandb, 
                                                      print_optuna_stats)
from debeir.training.hparm_tuning.trainer import SentenceTransformerHparamTrainer
from debeir.training.hparm_tuning.config import HparamConfig

# Load a hyper-parameter configuration file with optuna parameters
hparam_config = HparamConfig.from_json(
   "./configs/hparam/trec2021_tuning.json"
)

trainer = SentenceTransformerHparamTrainer(
   dataset_loading_fn=data_loading_fn,
   evaluator_fn=evaluation.BinaryClassificationEvaluator,
   hparams_config=hparam_config,
)

# Run optuna with wandb integration
study = run_optuna_with_wandb(trainer, wandb_kwargs={
    "project": "my-hparam-tuning-project"
})

# Print optuna stats and best run
print_optuna_stats(study)
```

More information on the library is found on the github page, <a href="https://www.github.com/ayuei/debeir" target="_blank"> DeBeIR </a>. Any feedback and suggestions are welcome at <a href="https://www.github.com/ayuei/debeir/issues" target="_blank"> issues </a>.

# References
