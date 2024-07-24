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
- **Phase 1 WER:** 34%
- **Phase 2 WER and DER:** 32% - 11%

## Model Details 

### ASR Model Enhancement 

We have entered the second phase of the competition, focusing on enhancing our Automatic Speech Recognition (ASR) model. Below are the key steps and improvements we have implemented:

*Enhancements*

**1- Tokenizer Enhancement:**

Upgraded the tokenizer to tokenizer_spe_bpe_v128, which features 128 vocabulary size, SentencePiece Encoding (SPE), and Byte Pair Encoding (BPE) for better handling of speech data.

**2- Configuration Parameter Optimization:**

Enhanced the configuration parameters:
-Increased batch size to accelerate training while ensuring accuracy remains unaffected.
-Enabled resuming training from the last best checkpoint. [Link to .yaml configuration file]

**3- Extended Training Duration:**

Increased the number of training epochs to 170 to improve model performance and convergence.

**4- Language Model Integration:**

Incorporated a language model to enhance recognition accuracy and contextual understanding. [Link to attached notebook]

-These enhancements have significantly improved our ASR model's performance, bringing us closer to achieving our project goals.

For more details here is the link for the [first phase Repo](https://github.com/Alkholy53/ASR-Squad).

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

# bestttttt
# This YAML file is created for all types of offline speaker diarization inference tasks in `<NeMo git root>/example/speaker_tasks/diarization` folder.
# The inference parameters for VAD, speaker embedding extractor, clustering module, MSDD module, ASR decoder are all included in this YAML file. 
# All the keys under `diarizer` key (`vad`, `speaker_embeddings`, `clustering`, `msdd_model`, `asr`) can be selectively used for its own purpose and also can be ignored if the module is not used.
# The configurations in this YAML file is suitable for telephone recordings involving 2~8 speakers in a session and may not show the best performance on the other types of acoustic conditions or dialogues.
# An example line in an input manifest file (`.json` format):
# {"audio_filepath": "/path/to/audio_file", "offset": 0, "duration": null, "label": "infer", "text": "-", "num_speakers": null, "rttm_filepath": "/path/to/rttm/file", "uem_filepath": "/path/to/uem/file"}
name: &name "ClusterDiarizer"

num_workers: 1
sample_rate: 16000
batch_size: 64
device: null # can specify a specific device, i.e: cuda:1 (default cuda if cuda available, else cpu)
verbose: True # enable additional logging

diarizer:
  manifest_filepath: ???
  out_dir: ???
  oracle_vad: False # If True, uses RTTM files provided in the manifest file to get speech activity (VAD) timestamps
  collar: 0.25 # Collar value for scoring
  ignore_overlap: True # Consider or ignore overlap segments while scoring

  vad:
    model_path: vad_multilingual_marblenet # .nemo local model path or pretrained VAD model name 
    external_vad_manifest: null # This option is provided to use external vad and provide its speech activity labels for speaker embeddings extraction. Only one of model_path or external_vad_manifest should be set

    parameters: # Tuned parameters for CH109 (using the 11 multi-speaker sessions as dev set) 
      window_length_in_sec: 0.15  # Window length in sec for VAD context input 
      shift_length_in_sec: 0.01 # Shift length in sec for generate frame level VAD prediction
      smoothing: "median" # False or type of smoothing method (eg: median)
      overlap: 0.5 # Overlap ratio for overlapped mean/median smoothing filter
      onset: 0.1 # Onset threshold for detecting the beginning and end of a speech 
      offset: 0.1 # Offset threshold for detecting the end of a speech
      pad_onset: 0.1 # Adding durations before each speech segment 
      pad_offset: 0 # Adding durations after each speech segment 
      min_duration_on: 0 # Threshold for small non_speech deletion
      min_duration_off: 0.2 # Threshold for short speech segment deletion
      filter_speech_first: True 

  speaker_embeddings:
    model_path: titanet_large # .nemo local model path or pretrained model name (titanet_large, ecapa_tdnn or speakerverification_speakernet)
    parameters:
      window_length_in_sec: [1.5,1.25,1.0,0.75,0.5] # Window length(s) in sec (floating-point number). either a number or a list. ex) 1.5 or [1.5,1.0,0.5]
      shift_length_in_sec: [0.75,0.625,0.5,0.375,0.25] # Shift length(s) in sec (floating-point number). either a number or a list. ex) 0.75 or [0.75,0.5,0.25]
      multiscale_weights: [1,1,1,1,1] # Weight for each scale. should be null (for single scale) or a list matched with window/shift scale count. ex) [0.33,0.33,0.33]
      save_embeddings: True # If True, save speaker embeddings in pickle format. This should be True if clustering result is used for other models, such as `msdd_model`.
  
  clustering: 
    parameters:
      oracle_num_speakers: False # If True, use num of speakers value provided in manifest file.
      max_num_speakers: 8 # Max number of speakers for each recording. If an oracle number of speakers is passed, this value is ignored.
      enhanced_count_thres: 80 # If the number of segments is lower than this number, enhanced speaker counting is activated.
      max_rp_threshold: 0.25 # Determines the range of p-value search: 0 < p <= max_rp_threshold. 
      sparse_search_volume: 30 # The higher the number, the more values will be examined with more time. 
      maj_vote_spk_count: False  # If True, take a majority vote on multiple p-values to estimate the number of speakers.
      chunk_cluster_count: 50 # Number of forced clusters (overclustering) per unit chunk in long-form audio clustering.
      embeddings_per_chunk: 10000 # Number of embeddings in each chunk for long-form audio clustering. Adjust based on GPU memory capacity. (default: 10000, approximately 40 mins of audio) 
  
  msdd_model:
    model_path: diar_msdd_meeting # .nemo local model path or pretrained model name for multiscale diarization decoder (MSDD)
    parameters:
      use_speaker_model_from_ckpt: True # If True, use speaker embedding model in checkpoint. If False, the provided speaker embedding model in config will be used.
      infer_batch_size: 25 # Batch size for MSDD inference. 
      sigmoid_threshold: [0.7] # Sigmoid threshold for generating binarized speaker labels. The smaller the more generous on detecting overlaps.
      seq_eval_mode: False # If True, use oracle number of speaker and evaluate F1 score for the given speaker sequences. Default is False.
      split_infer: True # If True, break the input audio clip to short sequences and calculate cluster average embeddings for inference.
      diar_window_length: 50 # The length of split short sequence when split_infer is True.
      overlap_infer_spk_limit: 5 # If the estimated number of speakers are larger than this number, overlap speech is not estimated.
  
  asr:
    model_path: stt_en_conformer_ctc_small # Provide NGC cloud ASR model name. stt_en_conformer_ctc_* models are recommended for diarization purposes.
    parameters:
      asr_based_vad: False # if True, speech segmentation for diarization is based on word-timestamps from ASR inference.
      asr_based_vad_threshold: 1.0 # Threshold (in sec) that caps the gap between two words when generating VAD timestamps using ASR based VAD.
      asr_batch_size: null # Batch size can be dependent on each ASR model. Default batch sizes are applied if set to null.
      decoder_delay_in_sec: null # Native decoder delay. null is recommended to use the default values for each ASR model.
      word_ts_anchor_offset: null # Offset to set a reference point from the start of the word. Recommended range of values is [-0.05  0.2]. 
      word_ts_anchor_pos: "start" # Select which part of the word timestamp we want to use. The options are: 'start', 'end', 'mid'.
      fix_word_ts_with_VAD: False # Fix the word timestamp using VAD output. You must provide a VAD model to use this feature.
      colored_text: False # If True, use colored text to distinguish speakers in the output transcript.
      print_time: True # If True, the start and end time of each speaker turn is printed in the output transcript.
      break_lines: False # If True, the output transcript breaks the line to fix the line width (default is 90 chars)
    
    ctc_decoder_parameters: # Optional beam search decoder (pyctcdecode)
      pretrained_language_model: null # KenLM model file: .arpa model file or .bin binary file.
      beam_width: 32
      alpha: 0.5
      beta: 2.5

    realigning_lm_parameters: # Experimental feature
      arpa_language_model: null # Provide a KenLM language model in .arpa format.
      min_number_of_words: 3 # Min number of words for the left context.
      max_number_of_words: 10 # Max number of words for the right context.
      logprob_diff_threshold: 1.2  # The threshold for the difference between two log probability values from two hypotheses.```
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

You can run this notebook on [kaggel](https://www.kaggle.com/code/abdallahmohamed53/final-diar-asr)


### What Parameters Can You Change in the future?

#### Types of Parameters

1. **Model Paths:** Specify different pre-trained model paths.
2. **Clustering Threshold:** Adjust the distance threshold for clustering.
3. **VAD Parameters:** Modify window length and hop length for feature extraction.


## Discussion & Results

### Numerical Results

| Number of Epochs | Word Error Rate (WER) |
|------------------|-----------------------|
| 5                | 73%                   |
| 30               | 44%                   |
| 100              | 34%                   |
| 170              | 32%                   |


### Future Work

To further improve the model, we plan to focus on the following areas:

1. **Increased Training Duration**:
    - Conduct additional training with more epochs, which will require more resources such as GPUs and extended training hours.

2. **Language Model Enhancement**:
    - Further refine the language model to improve contextual accuracy and overall performance.

3. **Parameter Optimization**:
    - Continue to enhance and fine-tune the model parameters for optimal performance.
