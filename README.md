# Egy_Speaker_Diarization
![lol](https://github.com/Alkholy53/Egy_Speaker_Diarization/blob/main/Images/asr_sd_diagram.png)
## Outline
- [1. Abstract](#abstract)
- [2. Model Details](#model-details)
    - [ASR Model Enhancement](#ASR-Model-Enhancement)
    - [Speaker Diarization Architecture and Models](#Speaker-Diarization-Architecture-and-Models)
- [3. Experiment Technicalities](#experiment-technicalities)
- [4. How to Use the Model?](#how-to-use-the-model)
- [5. Discussion & Results](#discussion--results)

## 1. Abstract

### Brief
This project focuses on Arabic Speech Recognition for Egyptian Dialects, specifically in noisy environments and multi-speaker scenarios. The ASR system has been trained on 100 hours of Egyptian dialect speech data and evaluated in two phases. The second phase involves diarization of multi-speaker recordings.

### Objective
The objective in this phase  to improve an ASR system that can accurately transcribe Egyptian Arabic speech and provide diarization of multi-speaker audio recordings, including speaker identification and timestamps.

### Numerical Result
- **Phase 1 Mean Levenshtein Distance:** 21
- **Phase 2 Mean Levenshtein Distance:** 

## 2. Model Details


## ASR Model Enhancement 

We have entered the second phase of the competition, focusing on enhancing our Automatic Speech Recognition (ASR) model. Attached in this [notebook](https://github.com/Alkholy53/Egy_Speaker_Diarization/blob/main/trainn_ASR_model.ipynb) the whole training process. Below are the key steps and improvements we have implemented:

### *Enhancements:*

**1- Tokenizer Enhancement:**

Upgraded the tokenizer to tokenizer_spe_bpe_v128, which features 128 vocabulary size, SentencePiece Encoding (SPE), and Byte Pair Encoding (BPE) for better handling of speech data.

**2- Configuration Parameter Optimization:**

Enhanced the configuration parameters:
-Increased batch size to accelerate training while ensuring accuracy remains unaffected.
-Enabled resuming training from the last best checkpoint. [configuration file](https://github.com/Alkholy53/Egy_Speaker_Diarization/blob/main/conformer_ctc_bpe.yaml)

**3- Extended Training Duration:**

Increased the number of training epochs to 170 to improve model performance and convergence.

**4- Language Model Integration:**

Incorporated a language model to enhance recognition accuracy and contextual understanding. [Link to attached notebook]

-These enhancements have significantly improved our ASR model's performance, bringing us closer to achieving our project goals.

For more details here is the link for the [first phase Repo](https://github.com/Alkholy53/ASR-Squad).

## Speaker Diarization Architecture and Models

The speaker diarization system integrates several sophisticated models to accurately segment and identify speakers in multi-speaker audio recordings.

#### 1. Speaker Embeddings Model: `titanet_large`

- **Model Name:** `titanet_large`
- **Description:** `titanet_large` is a pre-trained deep neural network model designed to extract speaker embeddings from audio segments. Speaker embeddings are high-dimensional vector representations that capture the unique characteristics of a speaker's voice. These embeddings are crucial for distinguishing between different speakers in a recording.
- **Architecture:** The `titanet_large` model is based on the state-of-the-art TiTANet architecture, which uses a combination of convolutional and recurrent layers to learn robust speaker features.
- **Usage:** The model processes short audio segments (typically 1-2 seconds) and outputs speaker embeddings. These embeddings are then used for clustering and speaker identification.
#### 2. Speaker Diarization Model: `diar_msdd_meeting`

- **Model Name:** `diar_msdd_meeting`
- **Description:** `diar_msdd_meeting` is a neural network model specifically designed for meeting scenarios, where multiple speakers may talk simultaneously or in quick succession. This model is capable of performing multi-speaker diarization, which involves identifying and segmenting each speaker in the audio.
- **Architecture:** The model uses a multi-scale diarization approach, combining short-term and long-term context information to accurately detect speaker changes and overlapping speech. The architecture includes layers optimized for capturing the temporal dynamics of conversations.
- **Usage:** The model takes speaker embeddings and audio features as input and produces speaker labels and timestamps for each segment of speech. It is particularly effective in handling overlapping speech and noisy environments.
### YAML Configuration File

Here is the [config.yaml](https://github.com/Alkholy53/Egy_Speaker_Diarization/blob/main/diar_infer_meeting.yaml) file, including the configurations for both the `titanet_large` and `diar_msdd_meeting` models.

## 3. Experiment Technicalities

### Components Overview (Big Picture)
![pipeline](https://github.com/Alkholy53/Egy_Speaker_Diarization/blob/main/Images/sd_pipeline.png)
The system comprises multiple components working in tandem to achieve high-accuracy ASR and diarization:

- **Component 1:** Speaker Embeddings Extraction (`titanet_large`)
- **Component 2:** Speaker Diarization (`diar_msdd_meeting`)
- **Component 3:** Clustering Algorithm (Agglomerative Hierarchical Clustering)
- **Component 4:** Voice Activity Detection (VADNet)

### Component 1: `titanet_large`
- **Function:** Extracts high-dimensional speaker embeddings from audio segments.
- **Details:** Uses a combination of convolutional and recurrent layers to capture speaker-specific features.

### Component 2: `diar_msdd_meeting`
- **Function:** Performs multi-speaker diarization using speaker embeddings and audio features.
- **Details:** Uses a multi-scale diarization approach to handle overlapping speech and noisy environments.

### Component 3: Clustering Algorithm
- **Function:** Clusters speaker embeddings to identify distinct speakers in the audio.
- **Details:** Uses Agglomerative Hierarchical Clustering with a specified distance threshold.

### Component 4: Voice Activity Detection (VADNet)
- **Function:** Detects voice activity in the audio to segment speech and non-speech regions.
- **Details:** Pre-trained VAD model to enhance diarization accuracy.

## 4. How to Use the Model?

### Utilization (Guide) - Process to Use the End-to-End Code

#### How to Make It Work

You can run this notebook on [kaggel](https://www.kaggle.com/code/abdallahmohamed53/final-diar-asr)


## 5. Discussion & Results


### What Parameters Can You Change in the future?

#### Types of Parameters

1. **Model Paths:** Specify different pre-trained model paths.
2. **Clustering Threshold:** Adjust the distance threshold for clustering.
3. **VAD Parameters:** Modify window length and hop length for feature extraction.


### Numerical Results

| Number of Epochs | Word Error Rate (WER) | Mean Levenshtein Distance |
|------------------|-----------------------|---------------------------|
| 5                | 73%                   |21                         |
| 30               | 44%                   |15                         |
| 100              | 34%                   |12.6                       |
| 170              | 32%                   |                           |

### Checkpoints

The ASR model checkpoints can be downloaded from the following link:

- [epoch=169-step=59840.ckpt](https://drive.google.com/file/d/1IYAYT4mKskn00-n7xHfnXP0udZqzhiNh/view?usp=drive_link)

### Future Work

To further improve the model, we plan to focus on the following areas:

1. **Increased Training Duration**:
    - Conduct additional training with more epochs, which will require more resources such as GPUs and extended training hours.

2. **Language Model Enhancement**:
    - Further refine the language model to improve contextual accuracy and overall performance.

3. **Parameter Optimization**:
    - Continue to enhance and fine-tune the model parameters for optimal performance.
