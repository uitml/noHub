# Hubs and Hyperspheres

This repository contains the code for the paper _"Hubs and Hyperspheres: Reducing Hubness and Improving Transductive Few-shot Learning with Hyperspherical Embeddings"_, submitted to CVPR 2023.

**Abstract:**
_Distance-based classification is frequently used in transductive few-shot learning (FSL). 
However, due to the high-dimensionality of image representations, FSL classifiers are prone to suffer from the hubness problem, where a few points (hubs) occur frequently in multiple nearest neighbour lists of other points. 
Hubness negatively impacts distance-based classification when hubs from one class appear often among the nearest neighbors of points from another class, degrading the classifier's performance. 
To address the hubness problem in FSL, we first prove that hubness can be eliminated by distributing representations uniformly on the hypersphere. 
We then propose two new approaches to embed representations on the hypersphere, which we prove optimize a tradeoff between uniformity and local similarity preservation -- reducing hubness while retaining class structure.
Our experiments show that the proposed methods reduce hubness, and significantly improves transductive FSL accuracy for a wide range of classifiers._


## Results aggregated over classifiers
**1-shot**

|              |                | mini      | mini     | tiered    | tiered   | CUB       | CUB      |
|:-------------|:---------------|:----------|:---------|:----------|:---------|:----------|:---------|
|              | Embedding      | Acc       | Score    | Acc       | Score    | Acc       | Score    |
| ResNet18     | None           | 55.74     | 0.17     | 62.61     | 0.0      | 63.78     | 0.17     |
|              | L2             | 68.22     | 2.33     | 75.94     | 2.17     | 78.09     | 2.33     |
|              | Cl2            | 69.56     | 2.83     | 76.97     | 3.0      | 78.26     | 2.83     |
|              | ZN             | 60.0      | 2.33     | 66.21     | 2.5      | 67.43     | 2.67     |
|              | ReRep          | 60.76     | 4.0      | 67.07     | 3.67     | 69.6      | 4.17     |
|              | EASE           | 69.63     | 3.67     | 77.05     | 4.0      | 78.84     | 3.67     |
|              | TCPR           | 69.97     | 4.0      | 77.18     | 3.33     | 78.83     | 4.0      |
|              | noHub (Ours)   | **72.58** | **6.83** | **79.77** | **6.83** | **81.91** | **6.83** |
|              | noHub-S (Ours) | **73.64** | **7.67** | **80.6**  | **7.67** | **83.1**  | **7.67** |
|              |                |           |          |           |          |           |          |
| WideRes28-10 | None           | 63.59     | 1.0      | 71.29     | 0.83     | 79.23     | 1.17     |
|              | L2             | 74.3      | 3.0      | 76.19     | 2.67     | 88.61     | 3.5      |
|              | Cl2            | 71.32     | 1.33     | 75.17     | 2.0      | 88.52     | 3.33     |
|              | ZN             | 64.27     | 2.5      | 65.64     | 2.5      | 76.0      | 1.5      |
|              | ReRep          | 65.51     | 3.0      | 71.83     | 3.17     | 83.1      | 3.5      |
|              | EASE           | 74.95     | 4.33     | 76.59     | 3.67     | 88.51     | 3.5      |
|              | TCPR           | 75.64     | 4.83     | 76.51     | 4.0      | 88.22     | 2.5      |
|              | noHub (Ours)   | **78.22** | **7.0**  | **79.76** | **7.0**  | **90.25** | **5.67** |
|              | noHub-S (Ours) | **79.24** | **7.67** | **80.46** | **7.67** | **90.82** | **7.67** |

**5-shot**

|              |                | mini      | mini     | tiered    | tiered   | CUB       | CUB      |
|:-------------|:---------------|:----------|:---------|:----------|:---------|:----------|:---------|
|              | Embedding      | Acc       | Score    | Acc       | Score    | Acc       | Score    |
| ResNet18     | None           | 69.83     | 0.83     | 74.38     | 0.67     | 76.01     | 1.17     |
|              | L2             | 81.58     | 2.33     | 86.05     | 1.83     | 88.43     | 2.83     |
|              | Cl2            | 81.95     | 2.67     | 86.43     | 3.0      | 88.49     | 2.5      |
|              | ZN             | 71.49     | 4.0      | 75.32     | 3.83     | 76.92     | 3.5      |
|              | ReRep          | 70.25     | 2.5      | 74.52     | 1.83     | 76.43     | 2.5      |
|              | EASE           | 81.84     | 3.5      | 86.4      | 3.17     | 88.57     | 3.5      |
|              | TCPR           | 82.1      | 4.0      | 86.54     | 3.83     | 88.79     | 4.33     |
|              | noHub (Ours)   | **82.58** | **5.5**  | **86.9**  | **4.5**  | **89.13** | **6.0**  |
|              | noHub-S (Ours) | **82.61** | **6.5**  | **87.13** | **6.67** | **88.93** | **5.33** |
|              |                |           |          |           |          |           |          |
| WideRes28-10 | None           | 78.77     | 1.5      | 84.1      | 1.67     | 89.49     | 1.67     |
|              | L2             | 85.65     | 4.0      | 86.29     | 3.83     | 93.47     | 3.67     |
|              | Cl2            | 83.14     | 1.33     | 85.47     | 1.5      | 93.49     | 4.0      |
|              | ZN             | 74.61     | 4.33     | 75.34     | 5.0      | 81.02     | 3.17     |
|              | ReRep          | 73.86     | 1.83     | 81.51     | 1.67     | 87.2      | 2.0      |
|              | EASE           | 85.51     | 3.5      | 86.29     | 3.33     | 93.34     | 3.5      |
|              | TCPR           | **86.03** | **6.0**  | 86.37     | 4.0      | 93.3      | 3.0      |
|              | noHub (Ours)   | **86.44** | **5.67** | **87.07** | **5.5**  | **93.65** | **4.17** |
|              | noHub-S (Ours) | 85.95     | 5.5      | **87.05** | **5.83** | **93.76** | **5.0**  |



## Datasets
The datasets can be downloaded by following the instructions in the repo [Realistic evaluation of transductive few-shot learning
](https://github.com/oveilleux/Realistic_Transductive_Few_Shot).

After downloading the datasets, use the files in `data/split` to separate the images into directories `data/[mini|tiered|cub]/[train|val|test]`.

## Feature extractors
Download the checkpoints from:
* ResNet-18: [Realistic evaluation of transductive few-shot learning](https://github.com/oveilleux/Realistic_Transductive_Few_Shot).
* WideRes28-10: [S2M2 Charting the Right Manifold: Manifold Mixup for Few-shot Learning](https://github.com/nupurkmr9/S2M2_fewshot)

And place them in the `models` directory.

### Feature caching
When executing the evaluation script, setting `--cache_dir /path/to/cached/features` will save the computed features to the specified directory.
By setting `--use_cached True` in subsequent runs, this will make repeated evaluations on the same dataset/feature extractors much faster.

## Installing dependencies
Run
```bash
conda env create -f environment.yml
```
to create a conda environment with the base requirements.

Then, inside the newly created environment, install the other dependencies with `pip`:
```bash
pip install -r requirements.txt
```

### Running with Docker
The `docker` directory contains a build script and a Dockerfile to build a docker image with all required dependencies.

## Running evaluation
Evaluation is performed by running
```bash
python evaluate.py --checkpoint "<path/to/feature/extractor/checkpoint.ckpt>" --n_shots "<shots>" --dataset "[mini|tiered|cub]" --classifier "<classifier>" --embedding "<embedding>" "<optional arguments>" 
```
where `"<classifier>"` and `"<embedding>"` are one of the implemented classifiers and embedding methods, respectively (see below).

Alternatively, the arguments can be provided in a `yaml` file:
```bash
python evaluate.py -c path/to/config/file.yml "<optional arguments>"
```
See `src/config/templates` for examples of config files.

### Implemented embedding methods

| Name                                                                                                                                    | `--embedding` |
|:----------------------------------------------------------------------------------------------------------------------------------------|:--------------|
 | None                                                                                                                                    | none          |
 | [L2](https://github.com/mileyan/simple_shot)                                                                                            | l2            |
 | [CL2](https://github.com/mileyan/simple_shot)                                                                                           | cl2           |
 | [ZN](https://openaccess.thecvf.com/content/ICCV2021/papers/Fei_Z-Score_Normalization_Hubness_and_Few-Shot_Learning_ICCV_2021_paper.pdf) | zn            |
 | [ReRep](https://proceedings.mlr.press/v139/cui21a.html)                                                                                 | rr            |
 | [EASE](https://github.com/allenhaozhu/EASE)                                                                                             | ease          |
 | [TCPR](https://arxiv.org/abs/2210.16834)                                                                                                | tcpr          |
| noHub (Ours)                                                                                                                            | nohub         | 
 | noHub-S (Ours)                                                                                                                          | nohubs        |
 
### Implemented classifiers
| Name                                                                  | `--classifier` |
|:----------------------------------------------------------------------|:---------------|
| [SmpleShot](https://github.com/mileyan/simple_shot)                   | simpleshot     |
| [Laplacianshot](https://github.com/imtiazziko/LaplacianShot)          | laplacianshot  |
| [α-TIM](https://github.com/oveilleux/Realistic_Transductive_Few_Shot) | alpha_tim      |
| [iLPC](https://github.com/MichalisLazarou/iLPC)                       | ilpc          |
| [Oblique Manifold](https://github.com/GuodongQi/FSL-OM)               | om            |
| [SIAMESE](https://github.com/allenhaozhu/EASE)                        | siamese       |

