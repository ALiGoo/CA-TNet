Learning and consolidating the contextualized contour representations of tones from F0 sequences and durational variations via transformers
===============
This repository contains our implementation of the paper, "Learning and consolidating the contextualized contour representations of tones from F0 sequences and durational variations via transformers". ([Paper link here]()).


## Installation
First, clone the repository locally and install the requirements :
```
$ git clone https://github.com/ALiGoo/CA-TNet.git
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
$ pip install -r requirements.txt
```


## Experiments

### Dataset
We released some sample data from the FCU-VOICE-255 and SINICA-MCDC-8 dataset to demonstrate how to run the code.

### How to Run
To evaluate the model on FCU-VOICE-255:
```
python main.py --config_file='configs/FCU-VOICE-255_config.yml'
```
To evaluate the model on SINICA-MCDC-8:
```
python main.py --config_file='configs/SINICA-MCDC-8_config.yml'
```

## Citation
If you use CA-TNet code in your research please use the following citation:

```bibtex

```

