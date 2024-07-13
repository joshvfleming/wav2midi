# wav2midi

This repo contains a reproduction of the model wav2midi model described in the following papers:

- [Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset](https://arxiv.org/abs/1810.12247)
- [Onsets and Frames: Dual-Objective Piano Transcription](https://arxiv.org/abs/1710.11153)

The implementation is also inspired by [https://github.com/jongwook/onsets-and-frames](https://github.com/jongwook/onsets-and-frames).

### Running

First, download and unzip the [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro) to a local folder, which we'll call `./maestro` here. You'll need just over 100gb to store it. This dataset contains hundreds of hours of recorded piano performances aligned with MIDI performance data.

#### Install dependencies
```shell
$ pip install -r requirements.txt
```

#### Run the data processing script
This script converts the data into our training set:
```shell
$ scripts/process-dataset.sh ./maestro ./data
```

#### Train the model
With default `batch_size` and `model_complexity` settings, this requires a GPU with at least 18gm of memory. You might need to play with these parameters to make it work on your hardware.
```shell
$ python train.py ./data ./out
```

#### Inference
The training process will produce model checkpoints in the `./out` folder. Let's use one to run inference on a recording:
```shell
$ python inference.py out/model-9.pt my_piano_recording.mp3
```

Inference will produce a MIDI file named `my_piano_recording.mid` in the same folder.
