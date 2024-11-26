from lenet import LeNet5
import copy
from masked_dem.dem_modules.components.dummy_tokenizer import DummyTokenizer
from masked_dem.energies.base_energy_function import BaseEnergyFunction
import hydra
from masked_dem.diffusion import Diffusion
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random

data_root = '/network/scratch/j/jarrid.rector-brooks/data/mnist'
random.seed(13)
torch.manual_seed(13)
data_train = MNIST(data_root,
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
data_test = MNIST(data_root,
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))
data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

net = LeNet5()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=2e-3)

def noise_images(images, diffusion):
    images = images.long()

    t = diffusion._sample_t(len(images), images.device)

    if diffusion.T > 0:
        t = (t * diffusion.T).to(torch.int)
        t = t / diffusion.T
        # t \in {1/T, 2/T, ..., 1}
        t += 1 / diffusion.T

    if diffusion.change_of_variables:
        unet_conditioning = t[:, None]
        f_T = torch.log1p(-torch.exp(-diffusion.noise.sigma_max))
        f_0 = torch.log1p(-torch.exp(-diffusion.noise.sigma_min))
        move_chance = torch.exp(f_0 + t * (f_T - f_0))
        move_chance = move_chance[:, None]
    else:
        sigma, dsigma = diffusion.noise(t)
        unet_conditioning = sigma[:, None]
        move_chance = 1 - torch.exp(-sigma[:, None])

    orig_shape = copy.deepcopy(images.shape)
    images_masked = diffusion.q_xt(
        images.flatten(1, -1),
        move_chance,
    ).reshape(*orig_shape)

    return images_masked.float()

def train(epoch, diffusion):
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        optimizer.zero_grad()

        output = net(noise_images(images, diffusion))

        loss = criterion(output, labels)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)

#         if i % 10 == 0:
#             print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

        loss.backward()
        optimizer.step()


def test():
    net.eval()
    with torch.no_grad():
        total_correct = 0
        avg_loss = 0.0
        for i, (images, labels) in enumerate(data_test_loader):
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(data_test)
    acc = float(total_correct) / len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), acc))
    return acc


def train_and_test(epoch, diffusion):
    print('training...')
    import pdb; pdb.set_trace()
    train(epoch, diffusion)
    acc = test()
    return acc


@hydra.main(version_base=None, config_path="../../masked_dem/configs", config_name="config")
def main(config):
    energy_function = hydra.utils.instantiate(config.energy_function)
    if not isinstance(energy_function, BaseEnergyFunction):
        energy_function = energy_function(tokenizer=tokenizer)

    tokenizer = DummyTokenizer(energy_function.vocab)

    diffusion = Diffusion(config, tokenizer=tokenizer, energy_function=energy_function)
    for e in range(1, 13):
        acc = train_and_test(e, diffusion)
        if e % 2 == 0:
            torch.save(net.state_dict(), f'./lenet_epoch={e}_test_acc={acc:0.3f}.pth')



if __name__ == '__main__':
    main()
