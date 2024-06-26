# FaceRSA: RSA-Aware Facial Identity Cryptography Framework (AAAI 24)

## Getting Started
### Prerequisites
```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install pyyaml
```
### Pretrained Model
Please download the pre-trained model from the following link, and place it under `./pretrained_models/`.
| Path | Description
| :--- | :----------
|[FaceRSA](https://drive.google.com/file/d/1qGFP47Ng_jG402FShVZnfAN6SYOzpVfK/view?usp=drive_link)  | Our pre-trained FaceRSA model.
### Auxiliary Models and Latent Codes
In addition, we provide various auxiliary models and latent codes inverted by [e4e](https://github.com/omertov/encoder4editing).
| Path | Description
| :--- | :----------
|[FFHQ StyleGAN](https://drive.google.com/file/d/1pts5tkfAcWrg4TpLDu6ILF5wHID32Nzm/view?usp=sharing) | StyleGAN model pretrained on FFHQ taken from [rosinality](https://github.com/rosinality/stylegan2-pytorch) with 1024x1024 output resolution.
## Testing
### Inference
The results are saved in `./sample/anonymized/` and `./sample/recovered/`
#### Example of Using FaceRSA with random password
```bash
python inference.py
```
## Citation
```bash
@inproceedings{zhang2024facersa,
  title={FaceRSA: RSA-Aware Facial Identity Cryptography Framework},
  author={Zhang, Zhongyi and Wei, Tianyi and Zhou, Wenbo and Zhao, Hanqing and Zhang, Weiming and Yu, Nenghai},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={7},
  pages={7423--7431},
  year={2024}
}
```
Training code will be released soon.
