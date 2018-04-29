import scipy.misc 
import numpy as np
import torch
from torch.autograd import Variable
from model import *
import pickle

data = pickle.load(open('../dataset/images.pixiv2.tagged.pkl', 'rb'))
imgs, tags = zip(*data)
x = Variable(torch.from_numpy(np.array(imgs[:5]))).cuda()
code = Variable(torch.randn(5, 128)).cuda()
vae = VAE(
        Conv2d96x96Encoder(128),
        Conv2d96x96Decoder(128)).cuda()
vae.load_state_dict(torch.load('model_e70.pt'))
_, _, imgs = vae(x)
#imgs = vae.decoder(code)
#imgs = np.hstack(imgs.data.cpu().numpy().transpose(0,2,3,1))
#scipy.misc.imsave('sample.jpg', imgs)
#exit()
x = np.hstack(x.data.cpu().numpy().transpose(0,2,3,1))
imgs = np.hstack(imgs.data.cpu().numpy().transpose(0,2,3,1))
scipy.misc.imsave('sample.jpg', np.vstack([x, imgs]))
