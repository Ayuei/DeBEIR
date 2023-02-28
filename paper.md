---
title: 'DeBEIR: A Python Package for Dense Bi-Encoder Information Retrieval'
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
 - name: "Australian National University, School of Computing"
   index: 1
 - name: "Commonwealth Scientific and Industrial Research Organisation, Data61"
   index: 2
date: 4 October 2022
bibliography: paper.bib
---

# Summary
Information Retrieval (IR) is the task of retrieving documents given a query or information need. These documents are retrieved and ranked based on a relevance function or relevance model such as Best-Matching 25 (BM25) [@bm25]. Although deep learning has been successful in other computer science fields, such as computer vision with AlexNet [@alexnet] and Inception [@googlenet] and natural language processing with transformers [@orig-bert-2018; @liu:2019; @biobert]; success in information retrieval was limited due to comparisons against weak baselines [@yang2019critically]. However, in 2019 [@lin-neural-recantation], deep learning in information retrieval could surpass less computationally intensive keyword-based statistical models in terms of retrieval effectiveness, sparking the field of dense retrieval. Dense retrieval is the task of retrieving documents given a query or information need using a dense vector representation of the query and documents [@lin2021pretrained]. The dense vector representation is obtained by passing the query and documents through a neural network. The neural network is usually a pre-trained language model such as BERT [@orig-bert-2018] or RoBERTa [@roberta]. The dense query vector representation is then used to retrieve documents using a similarity function such as cosine similarity.

Unlike statistical learning, tuning deep learning retrieval methods is often costly and time-consuming. This cost makes it essential to efficently automate much of the training, tuning and evaluation processes.

We present DeBEIR a library for: (1) facilitating dense retrieval research, primarily focusing on bi-encoder dense retrieval where query and documents dense vectors are generated separately [@reimers2019], (2) expedited experimentation in dense retrieval research by reducing boilerplate code through an interchangeable pipeline API and code extendability through the inheritance of general classes; (3) abstractions for standard training loops and hyperparameter tuning from easy-to-define configuration files. 

DeBEIR is aimed at helping practitioners, researchers and data scientists experimenting with bi-encoders by providing them with dense retrieval methods that are easy to use out of the box but also have additional extendability for more nuanced research. Furthermore, our pipeline runs asynchronously to reduce I/O performance bottlenecks, facilitating faster experiments and research.

A brief summary of the pipeline is (\autoref{fig:training}):

1. Configuration based on Tom's Obvious Minimal Language (TOML) files; these are loaded in a class factory to create pipeline objects.

2. An executor object takes in a query builder object. The purpose of the query builder object is to define the mapping of the documents and which parts of the query to use for query execution.

3. The executor object asynchronously runs the queries.

4. Finally, an evaluator object uses the results to list metrics defined by a configuration file against an oracle test set.


This pipeline is condensed into a single class that can be built from a configuration file.

# Statement of Need
Dense retrieval has been popular in Information Retrieval since 2015 [@paccr; @drmm; @abcnn]. In the early 2000s, there was a slow down in retrieval effectiveness as there needed to be more robust baselines [@Armstrong:2009] when proposing new methods. This stagnation repeated with the rise of deep learning, where retrieval effectiveness was again compared against weaker baselines and was not shown significantly stronger than statistical models [@yang2019critically], such as a well-tuned BM25 model.

However, this was later recanted when transformer models could be used fine-tuned on Natural Language Inference tasks or Ms-Marco [@msmarco] as a cross-encoder [@lin-neural-recantation], significantly overtaking even the best BM25 models.

There are generally two classes of dense retrieval models for IR: (1) the cross-encoder, which encodes queries and documents at query time and (2) the bi-encoder, which can encode documents at index time and queries at query time. The cross-encoder is generally more effective than the bi-encoder model for retrieval [@lin2021pretrained]. However, this increased effectiveness requires a more substantial computation and can be a bottleneck in production systems. Therefore, a less expensive model such as BM25 is typically used to retrieve smaller candidate lists (first-stage retrieval) to be fed to second-stage retrieval re-ranking by a cross-encoder.

Although cross-encoders are more accurate than bi-encoders, bi-encoder are more effective than BM25 [@search-like-an-expert-2022] and are faster than cross-encoders. Therefore, a gap in the literature in IR is to replace BM25 first-stage retrieval with a bi-encoder or
otherwise used as the sole ranking system, without a second-stage re-ranker. However, current libraries do not address this use case because it requires integration with the indexing and querying pipeline of the search engine.

DeBEIR is a library that addresses this gap by facilitating bi-encoder research and provides base classes with flexible functionality through inheritance. While we provide cross-encoder re-rankers for feature completeness, the library's priority is facilitating bi-encoder research. The strength of bi-encoders lies in the offline indexing of dense vectors. These vectors can then be used for first-stage retrieval and potentially passed to a second-stage retrieval system such as a cross-encoder. Bi-encoders can be used as the sole retrieval system when there is a lack of training data [@search-like-an-expert-2022] and, therefore, can be more useful in areas such as biomedical IR, where training data is scarce. Cross-encoders, however, require large amounts of training data for effectiveness.

The DeBEIR library offers an API for commonly used functions for training, hyper-parameter tuning (\autoref{fig:training}) and evaluation of transformer-based models. The pipeline can be broken up into multiple stages: parsing, query building, query execution, serialization and evaluation (\autoref{fig:pipeline}). Furthermore, we package our caching mechanism for the expensive encoding operations to speed up the pipeline during repeated experimentation.

Although similar libraries exist, such as sentence-transformers [@reimers2019], and openNIR [@opennir], they have less of a focus on the early stages of the dense retrieval pipeline. This stage involves indexing the textual data from the corpora and indexing dense vector representations, which is only helpful for bi-encoder type models over the traditional cross-encoder and is thus not typically explored by other libraries. Other limitations include a lack of extendability that restrict the users' options for training customization (we provide base classes that can be inherited) or that the library is tailored to general-purpose machine learning rather than informational retrieval. Finally, these libraries have a limited caching mechanism, as cross-encoders typically does not require this capability as it is decoupled from the index. Bi-encoders can have queries cached at query time to make repeated query calls to the index significantly faster.

DeBEIR will help facilitate early-stage dense retrieval and rapid experimentation research with bi-encoders. It is also flexible enough for second-stage retrieval using cross-encoders from this library or other libraries. We will continue to improve this tool over time.

![Standard flow of the DeEBIR query/evaluation loop.\label{fig:pipeline}](pipeline.pdf){scale=0.5}

![Standard flow of the DeEBIR training loop.\label{fig:training}](training.pdf){scale=0.43}

# Acknowledgments
The DeBEIR library uses Sentence-Transformers, Hugging Face's transformers and Datasets, allRank, Optuna, Elasticsearch and Trectools python packages.

This search is supported by CSIRO Data61, an Australian Government agency through the Precision Medicine FSP program and the Australian Research Training Program. We extend thanks to Brian Jin (Data61) for providing a code review.

# Examples
### Pipeline
The pipeline is a single class that can be built from a configuration file. The configuration file is a TOML file that defines the pipeline stages and their parameters. The pipeline is built using a class factory that takes in the configuration file and creates the pipeline stages. The pipeline stages are then executed in order.

```python
from debeir.interfaces.pipeline import NIRPipeline
from debeir.interfaces.callbacks import (SerializationCallback,
                                        EvaluationCallback)
from debeir.evaluation import Evaluator

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

More information on the library is found on the GitHub page, [DeBEIR](https://www.github.com/ayuei/debeir). Any feedback and suggestions are welcome by opening a thread in [DeBEIR issues](https://www.github.com/ayuei/debeir/issues).

# References
