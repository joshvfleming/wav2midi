# wav2midi

This repo contains a reproduction of the model wav2midi model described in the following papers:

- [Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset](https://arxiv.org/abs/1810.12247)
- [Onsets and Frames: Dual-Objective Piano Transcription](https://arxiv.org/abs/1710.11153)

The implementation is also inspired by [https://github.com/jongwook/onsets-and-frames](https://github.com/jongwook/onsets-and-frames).

## Demo

The transcription this model produces captures much of the performance from the recording, but there are still some bugs with clipped and missing notes. Here's an example:
    - [Original recording](https://drive.google.com/file/d/191xMbfwhel2E8kbVLy4lQzDAzA3TN00w/view). Goldberg Variation 1 performed by Glenn Gould.
    - [Transcribed by wav2midi](https://drive.google.com/file/d/1Z84Yf5l8xB_bsHg8I26I3J4Wz104uRXl/view). For fun, now that we have a transcription, we can play back the piece with a completely different instrument, say a string section.
    - Or en [electric guitar](https://drive.google.com/file/d/1eTiLvHqmbbcrG8boJelk6jyxIsXwbxWg/view)?

## Running

### Get the data
First, download and unzip the [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro) to a local folder, which we'll call `./maestro` here. You'll need just over 100gb to store it. This dataset contains hundreds of hours of recorded piano performances aligned with MIDI performance data.

### Install dependencies
```shell
$ pip install -r requirements.txt
```

### Run the data processing script
This script converts the data into our training set:
```shell
$ scripts/process-dataset.sh ./maestro ./data
```

### Train the model
With default `batch_size` and `model_complexity` settings, this requires a GPU with at least 18gm of memory. You might need to play with these parameters to make it work on your hardware.
```shell
$ python train.py ./data ./out
```

### Inference
The training process will produce model checkpoints in the `./out` folder. Let's use one to run inference on a recording:
```shell
$ python inference.py out/model-9.pt goldberg1.mp3
```

Inference will produce a MIDI file named `goldberg1.mid` in the same folder.
