# Adaptive Y-Net from a Causal Representation Perspective

This is an addition to the [official implementation](https://github.com/vita-epfl/causalmotion) for the paper

**Towards Robust and Adaptive Motion Forecasting: A Causal Representation Perspective**
<br>
<a href="https://sites.google.com/view/yuejiangliu">Yuejiang Liu</a>,
<a href="https://www.riccardocadei.com">Riccardo Cadei</a>,
<a href="https://people.epfl.ch/jonas.schweizer/?lang=en">Jonas Schweizer</a>,
<a href="https://sherwinbahmani.github.io">Sherwin Bahmani</a>,
<a href="https://people.epfl.ch/alexandre.alahi/?lang=en/">Alexandre Alahi</a>
<br>
École Polytechnique Fédérale de Lausanne (EPFL)

Links: **[`Arxiv 11/2021`](https://arxiv.org/abs/2111.14820) | [`Video (7 min)`](https://drive.google.com/file/d/1Uo0Y0eHq4vI7wOxya4mJlxbAe3U4kMx6/view) | [`Spurious`](https://github.com/vita-epfl/causalmotion/tree/main/spurious) | [`Style`](https://github.com/vita-epfl/causalmotion/tree/main/style)**
<br>
*Under review. Abbreviated version at NeurIPS DistShift, 2021.*

TL;DR: incorporate causal invariance and structure into the design and training of motion forecasting models
* causal formalism of motion forecasting with three groups of latent variables
* causal (invariant) representations to suppress spurious features and promote robust generalization
* causal (modular) structure to approximate a sparse causal graph and facilitate efficient adaptation

<p align="left">
  <img src="docs/overview.png" width="800">
</p>

If you find this code useful for your research, please cite our paper:

```bibtex
@article{liu2021causalmotion,
  title={Towards Robust and Adaptive Motion Forecasting: A Causal Representation Perspective},
  author={Liu, Yuejiang and Cadei, Riccardo and Schweizer, Jonas and Bahmani, Sherwin and Alahi, Alexandre},
  journal={arXiv preprint arXiv:2111.14820},
  year={2021}
}
```
### Original Code

Based on the original [Y-net](https://arxiv.org/pdf/2012.01526.pdf) [repository](https://github.com/HarshayuGirase/Human-Path-Prediction/tree/master/ynet)

### Setup

Install PyTorch, for example using pip

```
pip install --upgrade pip
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Install dependencies
```
pip install -r requirements.txt
```

Build [ddf](https://github.com/theFoxofSky/ddfnet) dependency
```
cd ddf
python setup.py install
mv build/lib*/* .
```

### Dataset

Get the raw dataset, our filtered custom dataset and segmentation masks for SDD from the original Y-net authors
```
pip install gdown && gdown https://drive.google.com/uc?id=14Jn8HsI-MjNIwepksgW4b5QoRcQe97Lg
unzip sdd_ynet.zip
```

After unzipping the file the directory should have following structure:
```
sdd_ynet
├── dataset_raw
├── dataset_filter
│   ├── dataset_ped
│   ├── dataset_biker
│   │   ├── gap
│   │   └── no_gap
│   └── ...
└── ynet_additional_files
```

In addition to our custom datasets in sdd_ynet/dataset_filter, you can create custom datasets:
```
bash create_custom_dataset.sh
```

### Scripts

1. Train Baseline

```
bash run_train.sh
```

&nbsp;&nbsp;&nbsp;&nbsp;Our pretrained models can be downloaded from [google drive](https://drive.google.com/drive/folders/1HzHP2_Mg2bAlDV3bQERoGQU3PvijKQmU).

2. Zero-shot Evaluation

```
bash run_eval.sh
```

3. Low-shot Adaptation

```
bash run_fine_tune.sh
```

### Basic Results

Results of different methods for low-shot transfer across agent types and speed limits.

<img src="docs/fewshot.png" height="180"/>
