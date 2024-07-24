# Egy_Speaker_Diarization
## Outline
- [Abstract](#abstract)
- [Model Details](#model-details)
- [Experiment Technicalities](#experiment-technicalities)
- [How to Use the Model?](#how-to-use-the-model)
- [Discussion & Results](#discussion--results)

## Abstract

### Brief
This project focuses on Arabic Speech Recognition for Egyptian Dialects, specifically in noisy environments and multi-speaker scenarios. The ASR system has been trained on 100 hours of Egyptian dialect speech data and evaluated in two phases. The second phase involves diarization of multi-speaker recordings.

### Objective
The objective in this phase  to improve an ASR system that can accurately transcribe Egyptian Arabic speech and provide diarization of multi-speaker audio recordings, including speaker identification and timestamps.

### Numerical Result
- **Phase 1 WER:** [Insert WER result]
- **Phase 2 WER and DER:** [Insert WER and DER results]

## Model Details 
 asr model 
this are details  about our  asr model
we improve our ASR model by these steps 
1
2
3
build language  model
like notebook for language model 



### Speaker Diarization Architecture and Models

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

Below is the `config.yaml` file, including the configurations for both the `titanet_large` and `diar_msdd_meeting` models:

```yaml
# Configuration for Speaker Diarization System

link ymal file
## Experiment Technicalities

### Components Overview (Big Picture)
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

## How to Use the Model?

### Utilization (Guide) - Process to Use the End-to-End Code

#### How to Make It Work

You can run this notebook on kaggel
https://www.kaggle.com/code/abdallahmohamed53/final-diar-asr

### What Parameters Can You Change in the futut?

#### Types of Parameters

1. **Model Paths:** Specify different pre-trained model paths.
2. **Clustering Threshold:** Adjust the distance threshold for clustering.
3. **VAD Parameters:** Modify window length and hop length for feature extraction.

## future work  & Results


### Numerical Results

- **Phase 1 WER:** [Insert WER result]
- **Phase 2 WER and DER:** [Insert WER and DER results]

