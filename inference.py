import torch
import yaml
import warnings

from PIL import Image
from tqdm import tqdm

from options.train_options import TestOptions
from utils.function import *
from trainer import *

warnings.filterwarnings('ignore')

device = torch.device('cuda')

opts = TestOptions().parse()
config = yaml.load(open('./configs/' + opts.config + '.yaml', 'r'), Loader=yaml.FullLoader)

trainer = Trainer(config, opts)
trainer.to(device)
trainer.load_checkpoint("./pretrained_models/model.pth")

img_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
input_resize = transforms.Resize([256,256])

# Your images to be anonymized
raw_annotations = glob.glob(r'./sample/origin/*.jpg')
raw_annotations.sort()

with torch.no_grad():
    trainer.mapper.eval()
    for img_path in tqdm(raw_annotations):
        label = img_path.split('/')[-1]
        image = img_to_tensor(Image.open(img_path)).unsqueeze(0).to(device)
        input_resize = transforms.Resize([256,256])
        input = input_resize(image)
        ori_wcode = trainer.e4e.encoder(input)
        if trainer.e4e.opts.start_from_latent_avg:
            if ori_wcode.ndim == 2:
                ori_wcode = ori_wcode + trainer.e4e.latent_avg.repeat(ori_wcode.shape[0], 1, 1)[:, 0, :]
            else:
                ori_wcode = ori_wcode + trainer.e4e.latent_avg.repeat(ori_wcode.shape[0], 1, 1)
        ori_inversion_img, _ = trainer.StyleGAN([ori_wcode], input_is_latent=True, randomize_noise=False)
        p1, _, inv_p, r_inv_p, r_inv_p2 = generate_code(16, 1, device, inv=True, gen_random_WR=True)
        wcode_fake, fake = trainer.mapper(ori_wcode, p1, opts.start_layer, opts.mapping_layers)
        wcode_recon, recon = trainer.mapper(wcode_fake, inv_p, opts.start_layer, opts.mapping_layers)

        inv_img = tensor2im(ori_inversion_img[0])
        inv_img.save(f'./sample/inversion/{label}')        
        fake_img = tensor2im(fake[0])
        fake_img.save(f'./sample/anonymized/{label}')
        recon_img = tensor2im(recon[0])
        recon_img.save(f'./sample/recovered/{label}')



