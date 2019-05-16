import torch
import torch.nn.functional as F
import torch.nn.modules.loss as loss


# Reconstruction + KL divergence losses summed over all elements and batch
class ReconKLD(loss._WeightedLoss):

    def forward(self, output, _):
        x, recon_x, mu, logvar = output
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD