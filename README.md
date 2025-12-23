<h1 align="center">Relational-Group-Activity-Recognition</h1>

<p align="center">
This repository presents an implementation of the <strong>ECCV 2018 paper</strong>,
<a href="https://openaccess.thecvf.com/content_ECCV_2018/papers/Mostafa_Ibrahim_Hierarchical_Relational_Networks_ECCV_2018_paper.pdf">
    <em>Hierarchical Relational Networks for Group Activity Recognition</em>
  </a>.The work addresses the limitations of conventional feature aggregation strategies,such as max pooling, average pooling, and attention-based pooling, which compress individual representations into a global descriptor while often discarding critical spatial and relational information.

To overcome these limitations, the proposed approach introduces a relational layer that explicitly models inter-person interactions within a structured relational graph. Rather than treating individuals independently, each person’s representation is refined through learned relationships with neighboring actors, capturing both local interactions and higher-level group dynamics.

By hierarchically aggregating relational information from the person level to the scene level, the model enables richer contextual reasoning and more accurate group activity recognition, particularly in complex multi-agent environments.
</p>

---

## Table of Contents
1. [Key Updates](#key-updates)
2. [Introduction](#introduction)
   - [How the Relational Layer Works](#how-the-relational-layer-works)
3. [Usage](#usage)  
   - [Clone the Repository](#1-clone-the-repository)  
   - [Install Dependencies](#2-install-the-required-dependencies)  
   - [Download Model Checkpoint](#3-download-the-model-checkpoint)  
     - [Option 1: Use Python Code](#option-1-use-python-code)  
     - [Option 2: Download Directly](#option-2-download-directly)  
4. [Dataset Overview](#dataset-overview)  
   - [Example Annotations](#example-annotations)  
   - [Train-Test Split](#train-test-split)  
   - [Dataset Statistics](#dataset-statistics)  
     - [Group Activity Labels](#group-activity-labels)  
     - [Player Action Labels](#player-action-labels)  
   - [Dataset Organization](#dataset-organization)  
   - [Dataset Download Instructions](#dataset-download-instructions)  
5. [Ablation Study](#ablation-study)  
   - [Baselines](#baselines)  
     - [Single Frame Models](#single-frame-models)  
       - [Performance Comparison (Original Paper)](#performance-comparison)  
       - [My Scores (Accuracy)](#my-scores-accuracy)  
     - [Temporal Models](#temporal-models)  
       - [Performance Comparison (Original Paper)](#performance-comparison-1)  
       - [My Scores (Accuracy)](#my-scores-accuracy-1)  
     - [Attention Models (New Baseline)](#attention-models-new-baseline)
       - [My Scores (Accuracy)](#my-scores-accuracy-2)    
       - [Confusion Matrix](#rcrg-2r-11c-conc-v1-attention-confusion-matrix)

## Key Updates

- **ResNet-50 Backbone**: Replaces the original VGG-19 backbone with a deeper ResNet-50 architecture, enabling stronger and more discriminative visual feature representations.
- **Ablation Studies**: Includes systematic ablation experiments to quantify the contribution of individual architectural components and design choices.
- **Test-Time Augmentation (TTA)**: Incorporates test-time data augmentation to improve inference robustness and reduce prediction variance.
- **Graph Attention Operator**: Implements an attention-based relational layer that dynamically weighs inter-person interactions within the relational graph.
- **Improved Performance**: Achieves consistently higher recognition accuracy across all evaluated baselines compared to the original implementation reported in the paper.
- **Modern Implementation**: Fully implemented in **PyTorch**, with graph operations supported by **PyTorch Geometric**, facilitating extensibility and reproducibility.

---

# Introduction

Conventional pooling mechanisms (max, average, or attention pooling) aggregate features in a manner that reduces dimensionality but frequently discards fine-grained spatial and relational cues critical for understanding group dynamics. In contrast, the Hierarchical Relational Network (HRN) introduces a dedicated relational layer that explicitly captures inter-person interactions within a structured relationship graph, leading to richer and more discriminative group-level representations.

<p align="center">
  <img width="512" height="512" src="https://github.com/user-attachments/assets/639cb140-a4df-4cd4-befc-2e965030723c" alt="Relational Layer Illustration"/>
</p>

## How the Relational Layer Works

The relational layer explicitly models inter-person interactions by representing individuals as nodes in a structured graph and iteratively refining their representations through learned relational operators.

### 1. Graph Construction
- Each detected person within a frame is represented as a **node** in the graph.
- Individuals are ordered according to the top-left corner coordinates `(x, y)` of their bounding boxes (sorted by `x`, then by `y` in case of ties) to ensure deterministic graph construction.
- Edges connect each person to a predefined set of neighboring individuals, forming local **cliques** that capture spatially proximal interactions.

### 2. Initial Person Features
Each person’s initial feature representation is extracted using a CNN backbone (e.g., **ResNet-50**):

   $$P_i^0 = \text{CNN}(I_i)$$

   where $I_i$ denotes the cropped image region corresponding to person $i$. 

### 3. Relational Update

<p align="center">
  <img src="https://github.com/user-attachments/assets/d965ea0e-8599-4d0a-b20e-73c50fbfe6d0" alt="Relational Graph Illustration" width="750"/>
</p>

At relational layer $\ell$, the representation of person person $i$ is updated as:

   $$P_i^\ell = \sum_{j \in E_i^\ell} F^\ell(P_i^{\ell-1} \oplus P_j^{\ell-1}; \theta^\ell)$$
   
where:
- $E_i^\ell$: denotes the set of neighbors of person $i$ in graph $G^\ell$,
- $\oplus$: represents feature concatenation,
- $F^\ell$: is a shared multi-layer perceptron (MLP) parameterized by $\ell$, with input dimension $2N_{\ell-1}$ and output dimension $N_\ell$.

This operation computes pairwise relational features between person $i$ and its neighbors, which are subsequently aggregated to form an updated representation.

### 4. Hierarchical Stacking

<p align="center">
  <img src="https://github.com/user-attachments/assets/8d8f4ea7-803c-486d-8fb7-00638445ddb7" alt="Hierarchical Relational Layers" width="750"/>
</p>

- Multiple relational layers are stacked hierarchically, progressively compressing person-level features while enriching them with higher-order relational context.
- The architecture naturally supports a variable number of individuals $K$, making it robust to occlusions, missed detections, and false positives.

### 5. Scene Representation
The final scene-level representation $S$ is obtained by aggregating the person features from the last relational layer:

 $$S = P_1^L \ ▽ \ P_2^L \ ▽ \dots \ ▽ \ P_K^L$$

where $▽$ denotes a pooling operator, such as concatenation or element-wise max pooling.


-----
## Usage

---

### 1. Clone the Repository
```bash
git clone https://github.com/mostafabahaa25/Hierarchical-Relational-Networks-for-GARR
```

### 2. Install the Required Dependencies
```bash
pip3 install -r requirements.txt
```

-----
## Dataset Overview

The dataset was created using publicly available YouTube volleyball videos. The authors annotated 4,830 frames from 55 videos, categorizing player actions into 9 labels and team activities into 8 labels. 

### Example Annotations

![image](https://github.com/user-attachments/assets/50f906ad-c68c-4882-b9cf-9200f5a380c7)

- **Figure**: A frame labeled as "Left Spike," with bounding boxes around each player, demonstrating team activity annotations.

![image](https://github.com/user-attachments/assets/cca9447a-8b40-4330-a11d-dbc0feb230ff)

### Train-Test Split

- **Training Set**: 3,493 frames
- **Testing Set**: 1,337 frames

### Dataset Statistics

#### Group Activity Labels
| Group Activity Class | Instances |
|-----------------------|-----------|
| Right set            | 644       |
| Right spike          | 623       |
| Right pass           | 801       |
| Right winpoint       | 295       |
| Left winpoint        | 367       |
| Left pass            | 826       |
| Left spike           | 642       |
| Left set             | 633       |

#### Player Action Labels
| Action Class | Instances |
|--------------|-----------|
| Waiting      | 3,601     |
| Setting      | 1,332     |
| Digging      | 2,333     |
| Falling      | 1,241     |
| Spiking      | 1,216     |
| Blocking     | 2,458     |
| Jumping      | 341       |
| Moving       | 5,121     |
| Standing     | 38,696    |

### Dataset Organization

- **Videos**: 55, each assigned a unique ID (0–54).
- **Train Videos**: 1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54.
- **Validation Videos**: 0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51.
- **Test Videos**: 4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47.

### Dataset Download Instructions

1. Enable Kaggle's public API. Follow the guide here: [Kaggle API Documentation](https://www.kaggle.com/docs/api).  
2. Use the provided shell script:
```bash
  chmod 600 .kaggle/kaggle.json 
  chmod +x script/script_download_volleball_dataset.sh
  .script/script_download_volleball_dataset.sh
```
For further information about dataset, you can check out the paper author's repository:  
[link](https://github.com/mostafa-saad/deep-activity-rec)

---

## [Ablation Study](https://en.wikipedia.org/wiki/Ablation_(artificial_intelligence)#:~:text=In%20artificial%20intelligence%20(AI)%2C,resultant%20performance%20of%20the%20system)

### Baselines

#### Single Frame Models:

- **B1-NoRelations:** In the first stage, a ResNet-50 backbone is fine-tuned, and each person is represented using a 2048-dimensional feature vector. In the second stage, individual features are projected through a shared fully connected layer of 128 dimensions. The resulting person-level representations are then pooled and passed to a softmax classifier for group activity recognition.

- **RCRG-1R-1C:** A pretrained ResNet-50 backbone is fine-tuned to extract 2048-dimensional person features, followed by a single relational layer (1R). All individuals are connected within a single clique (1C), enabling the modeling of all-pairwise relationships.

- **RCRG-1R-1C-!tuned:** Identical to the previous variant, except that the pretrained ResNet-50 backbone is kept frozen and not fine-tuned during training.

- **RCRG-2R-11C:** Extends the RCRG-1R-1C model by stacking two relational layers (2R) with output dimensions of 256 and 128, respectively. Both layers use a single clique (1C) encompassing all individuals. This variant and the subsequent ones explore the impact of deeper relational hierarchies.

- **RCRG-2R-21C:** Similar to RCRG-2R-11C, but the first relational layer uses two cliques (2C), corresponding to one clique per team, while the second layer models all-pairwise relations using a single clique (1C).

- **RCRG-3R-421C:** Employs three relational layers with output dimensions of 512, 256, and 128, respectively. The clique configuration follows a (4, 2, 1) hierarchy: the first layer contains four cliques (each team split into two sub-cliques), the second layer contains two cliques, and the final layer aggregates all individuals into a single clique.

##### Performance comparison

###### Original Paper Baselines Score

<img width="615" height="542" alt="{98F2C621-4E89-47FD-A112-A25946D611F3}" src="https://github.com/user-attachments/assets/4f7cc2f7-2b6d-472d-9555-d4d9b2de65cc" />

###### My Scores (Accuracy)

| Model | Test Acc | Test Acc TTA (4) | Paper Test ACC |
| :---- | :---: | :---: | :---: |
| B1-no-relations | 89.07% | 89.06% | 85.1% |
| RCRG-1R-1C | 89.42% | \- | 86.5% |
| RCRG-1R-1C-untuned | 80.86% | \- | 75.4% |
| RCRG-2R-11C | 89.15% | \- | 86.1% |
| RCRG-2R-21C | 89.49% | \- | 87.2% |
| RCRG-3R-421C | 88.97% | \- | 86.4% |
| **RCRG-2R-11C-conc** | **89.60%** | **89.71%** | 88.3% |
| **RCRG-2R-21C-conc** | **89.60%** | 89.60% | 86.7% |
| RCRG-3R-421C-conc | 89.23% | \- | 87.3% |

Notes:
- The `-conc` suffix denotes the use of concatenation pooling instead of max pooling.
- Test-time augmentation (TTA) uses four spatial transformations.

  
#### Temporal Models:

- **RCRG-2R-11C-conc-temporal:** Uses two relational layers (2R) with output dimensions of 256 and 128, respectively. Both relational layers operate on a single clique (1C) containing all detected individuals, enabling full pairwise interaction modeling across time.

- **RCRG-2R-21C:** The first relational layer partitions individuals into two cliques (2C), corresponding to one clique per team, while the second relational layer models all-pairwise interactions using a single clique (1C).

##### Performance comparison

###### Original Paper Baselines Score

<img width="523" height="323" alt="{848262DC-9865-4F49-A7CA-60B08675A6B8}" src="https://github.com/user-attachments/assets/de4eb7cb-3f3c-4320-baef-c7679055a6dd" />

###### My Scores (Accuracy)

| Model | Test Acc | Test Acc TTA (3) | Paper Test ACC |
| :---- | :---: | :---: | :---: |
| B1-no-relations-temporal | 88.93% | 89.60% | \- |
| RCRG-2R-11C-conc-V1 | 90.50% | 90.73% | 89.5% |
| RCRG-2R-11C-conc-V2 | **91.55%** | 91.62% | 89.5% |
| RCRG-2R-11C-conc-V3 | 91.40% | **91.77%** | 89.5% |
| RCRG-2R-21C | \- | \- | 89.4% |

Notes:
- The `-temporal` suffix indicates models operating on sequences of frames rather than single-frame inputs.
- The `-conc` suffix denotes the use of concatenation pooling instead of max pooling.
- The original paper does not explicitly specify the integration point of the LSTM module within the architecture. To address this ambiguity, three temporal variants were implemented:
  - `V1`: LSTM **before** the relational layer, allowing the relational operator to learn richer spatio-temporal person representations.
  - `V2`: LSTM **after** the relational layer, enabling temporal modeling over relationally enhanced features.
  - `V3`: LSTMs **both before and after** the relational layer, combining the benefits of V1 and V2.
- Training was focused on the `RCRG-2R-11C-conc` configuration, as it consistently achieved the strongest performance in both the original paper and the current implementation.
- The `B1-no-relations-temporal` baseline was implemented to explicitly evaluate the impact of relational modeling in the temporal setting; this baseline was not included in the original paper.

#### Attention Models (new baseline):

- Uses two relational layers (2R) operating on a single clique (1C) containing all players, where the standard MLP-based relational operator is replaced with a **graph attention mechanism** to dynamically weight inter-person interactions.

###### My Scores (Accuracy)

| Model | Test Acc | Test Acc TTA (3) | Paper Test ACC |
| :---- | :---: | :---: | :---: |
| RCRG-2R-11C-conc-V1 | 91.77% | **92.00%** | \- |

###### RCRG-2R-11C-conc-V1-Attention Confusion Matrix

<img src="experiments/attention_models/RCRG-R2-C11-conc-temporal_V1_2025_06_28_18_37/Group_Activity_RCRG-R2-C11-Conc-Temporal-Attention-TTA_Eval_On_Testset_TTA_confusion_matrix.png" alt="RCRG-2R-11C-conc-V1" >

-----
