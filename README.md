# Microservice Fault Detection Using Multi-modal Learning

## Project Overview

A full-stack fault detection system for microservices using multi-modal machine learning approaches.


## Writer

Yifan Fu



## Research Background & Motivation

Traditional fault detection methods typically rely on single-modal data analysis, such as:
- System logs
- Call traces
- Time series data

However, single-modal approaches have significant limitations:
* Incomplete information capture
* Difficulty in detecting potential correlations
* Limited ability to identify complex fault patterns
* Reduced accuracy in comprehensive fault diagnosis

  
### Architecture Overview
<img width="1051" alt="image" src="https://github.com/user-attachments/assets/abfa960a-0924-425b-81f2-c2dffa696a7f" />

<img width="1294" alt="image" src="https://github.com/user-attachments/assets/ceec6818-730f-4307-88d0-cdbddc2d7893" />

<img width="1440" alt="image" src="https://github.com/user-attachments/assets/73f3daf2-8b3b-4c6b-9031-b59d6de7c140" />


## Tech Stack

### Frontend
- Vue.js 3
- TypeScript 
- Vuex for state management
- Element Plus UI
- ECharts for visualization
- Axios for API calls

### Backend
- Python 3.7+
- FastAPI
- PyTorch
- Deep Graph Library (DGL)
- BERT/GloVe for embeddings
- MongoDB for data storage

## Core Features

### 1. Multi-modal Data Processing
- Log parsing using Drain algorithm
- Time series feature extraction
- Call chain graph construction
- Data fusion pipeline:
 - Text modality (logs)
 - Temporal modality (time series)
 - Graph modality (call chains)

<img width="1627" alt="image" src="https://github.com/user-attachments/assets/387553ef-bf30-4884-9951-dd897c31b908" />

<img width="886" alt="image" src="https://github.com/user-attachments/assets/e5cd7dbb-4d35-4fdb-8536-5aa2155958fe" />

<img width="1164" alt="image" src="https://github.com/user-attachments/assets/a9574141-23ff-4a16-ae10-a0efc436c2ad" />



### 2. Model Architecture
- Semi-supervised learning with PU Learning
- Three key components:
 1. GGNN-DeepSVDD for initial training
 2. GraphSAGE for classification
 3. Online fault prediction system

<img width="1618" alt="image" src="https://github.com/user-attachments/assets/7fe5b512-54d1-4d5a-895c-ac0feae5cfde" />

<img width="1696" alt="image" src="https://github.com/user-attachments/assets/7e373b38-7aab-44a4-896f-8fb7741bfba7" />

<img width="592" alt="image" src="https://github.com/user-attachments/assets/7d2bd726-dd7a-48a9-9bfb-fb12e5ad10ef" />


### 3. Testing Framework
#### Model Variations
- PU Learning + DeepTraLog (Best F1-score: 98.67%)
- DeepTraLog(F1-score: 98.25%)
- CNN Network
- CNN Classifier (14 fault types)



 ## Experimental Results

Our multi-modal approach achieves significant improvements over single-modal baselines:

| Model | Data Modality | Precision | Recall | F1-score |
|-------|---------------|-----------|---------|-----------|
| PU Learning | Multi-modal | 99.98% | 97.38% | 98.67% |
| DeepTraLog | Multi-modal | 96.55% | 100.0% | 98.25% |
| CNN (Baseline) | Single-modal | 90.0% | 88.73% | 89.36% |


<img width="1072" alt="image" src="https://github.com/user-attachments/assets/6b732131-f6e5-4143-ba5b-2f216675f9bd" />
