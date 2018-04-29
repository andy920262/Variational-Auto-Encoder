import pickle
import argparse

import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from progressbar import ProgressBar, ETA, FormatLabel, Bar

from model import *
from loader import *

parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--encode-size', type=int, default=128)
args = parser.parse_args()

data = pickle.load(open('../dataset/images.pixiv2.tagged.pkl', 'rb'))
imgs, tags = zip(*data)

train_loader = DataLoader(
        dataset=ImageDataset(imgs),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8)

vae = VAE(
        Conv2d96x96Encoder(args.encode_size),
        Conv2d96x96Decoder(args.encode_size)).cuda()

optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)

for epoch in range(args.epochs):
    total_loss = []
    widgets = [FormatLabel(''), ' ',
            ETA(), ' ', FormatLabel('')]
    pbar = ProgressBar(widgets=widgets, maxval=len(train_loader))
    pbar.start()
    for batch_i, batch_imgs in enumerate(train_loader):
        batch_imgs = Variable(batch_imgs).cuda()
        loss = vae.loss(batch_imgs, *vae(batch_imgs))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.append(loss.data[0] / args.batch_size / (96 * 96 * 3))
        widgets[0] = FormatLabel('Epoch:{}, {}/{}'.format(epoch, batch_i, len(train_loader)))
        widgets[-1] = FormatLabel('train_loss:{:.4f}'.format(np.mean(total_loss)))
        pbar.update(batch_i)
    pbar.finish()
    if epoch % 10 == 0:
        torch.save(vae.state_dict(), open('model_e{}.pt'.format(epoch), 'wb+'))
        

