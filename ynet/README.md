# Y-net

### Setup

Environments

```
pip install --upgrade pip
pip install -r requirements.txt
pip3 install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

Pre-trained
```
pip install gdown && gdown https://drive.google.com/uc?id=1u4hTk_BZGq1929IxMPLCrDzoG3wsZnsa
cd -rf ynet_additional_files/* ./
```

Notebook

```
jupyter notebook --no-browser --port=8888
ssh -N -f -L localhost:8888:localhost:8888 <username>@128.178.91.177
```
