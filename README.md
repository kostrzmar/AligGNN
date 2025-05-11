# GNN Text-To-Graph Evaluator (GNN TTGE)

GNN TTGE , a graph-based approach that investigates the impact of graph representations on various NLP tasks to evaluate different graph
structures, combined with pre-trained Language Models (LMs) and linguistic information.


## Align GNN Toolkit

Module which evaluates similarity of text by processing a pair of texts represented as graphs. GNN TTGE uses a layered convolutional network and reads the processed graphs through a pooling layer structured in a Siamese architecture. The resulting graph embeddings are concatenated and fed into a scoring head to compute the final similarity score. During training, the model optimises its performance by minimising a weighted loss function that incorporates both the graph embeddings and the output from the scoring head.


## Benchmark generation 

Module which uses a contrastive misclassification refinement approach to generate benchmark dataset out of manual annotated datasets. 


## Text graph visualisation 

Module which visualises a text graph representation. 


### Installation 

In case you would like to use CoreNLP, follow installation [instruction](https://stanfordnlp.github.io/CoreNLP/download.html). 

Since the library uses [Torchtext](https://pytorch.org/text/stable/index.html) which is EOL (April 2024 was the last stable release of the library) and compatible with [pytorch (2.3)](https://github.com/pytorch/text/releases), follows installation steps: 

#### Install environment 

```
conda create -n gnn_ttge python=3.9
conda activate gnn_ttge
```
or 
```
python -3.9 -m venv gnn_ttge
source gnn_ttge/bin/activate
```

#### Install torch and torch-geometric

Check cuda version (i.e. nvidia-smi, etc.)

```
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu[CUDA_VERSION] --no-cache-dir
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.3.1+cu[CUDA_VERSION].html --no-cache-dir
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.3.1+cu[CUDA_VERSION].html --no-cache-dir
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu[CUDA_VERSION].html --no-cache-dir
```

#### Install dependencies
```
pip install GPUtil tabulate coloredlogs pyyaml ray "ray[tune]" datasets torchtext sentence_transformers diskcache stanfordcorenlp unidecode word2number amrlib stanza matplotlib seaborn mlflow nevergrad --no-cache-dir
```

### Execute experiments 


Prepare experiment configuration and store to experiments paths (some example: align_gnn_toolkit/experiments_repository/)

```
python align_gnn_toolkit/execute_experiments.py -conf PATH_TO_THE_EXPERIMENT_CONFIGURATION
```

```
python align_gnn_toolkit/execute_experiments.py  PATH_TO_THE_FOLDERS_WITH_EXPERIMENT_CONFIGURATIONS
```

