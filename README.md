# FaceRSA: RSA-Aware Facial Identity Cryptography Framework (AAAI 24)

## Getting Started
### Prerequisites
```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install pyyaml
```
### Pretrained Model
Please download the pre-trained model from the following link.
| Path | Description
| :--- | :----------
|[FaceRSA]()  | Our pre-trained FaceRSA model.
### Auxiliary Models and Latent Codes
In addition, we provide various auxiliary models and latent codes inverted by [e4e](https://github.com/omertov/encoder4editing).
| Path | Description
| :--- | :----------
|[FFHQ StyleGAN](https://drive.google.com/file/d/1pts5tkfAcWrg4TpLDu6ILF5wHID32Nzm/view?usp=sharing) | StyleGAN model pretrained on FFHQ taken from [rosinality](https://github.com/rosinality/stylegan2-pytorch) with 1024x1024 output resolution.
## Testing
### Inference
The results are saved in `./anonymized/` and `./recovered/`
#### Example of Using FaceRSA with random password
```bash
python inference.py
```
Training code will be released soon.
