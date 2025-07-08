import argparse
import torch
import numpy as np
import scanpy as sc
import os
import pickle
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import tensor as Tensor

# Import your custom cel module and model definitions
import celery as cel

class ClusterVAE(nn.Module):
	def __init__(self,  
				 # in_channels: int,
				 latent_dim: int,
				 total_cluster: int,
				 hidden: List = None,
				 fgx = 2, fgy = 2,
				 **kwargs) -> None:
		super(ClusterVAE, self).__init__()

		self.latent_dim = latent_dim
		self.total_cluster = total_cluster
		
		scanx = fgx % 4 + 3
		scany = fgy % 4 + 3
		
		if hidden is None:
			hidden = [16, 8, 4, 8, 8]
		
		self.hidden = hidden
		# encoder
		self.encoderl1 = nn.Sequential( # like the Composition layer you built
			nn.Conv2d(1, hidden[0], [scanx,scany]),  # 76,  116		   178 208   # 80, 86
			nn.LeakyReLU())
		# self.encoderl2 = nn.Sequential(nn.MaxPool2d(2, stride=2))  #38, 58		 # 78, 82
		self.encoderl3 = nn.Sequential(
			nn.Conv2d(hidden[0], hidden[1], 4, stride=2),
			nn.LeakyReLU())	# 18, 28         # 38, 40
		self.encoderl4 = nn.Sequential(
			nn.Conv2d(hidden[1], hidden[2], 4, stride=2),     #18, 19
			nn.LeakyReLU())	# 15, 25
		# decoder
		self.decoderl4 = nn.Sequential(
			nn.ConvTranspose2d(hidden[2], hidden[3], 4, stride=2),
			nn.LeakyReLU())	# 35, 54
		self.decoderl3 = nn.Sequential(
			nn.ConvTranspose2d(hidden[3], hidden[4], 4, stride=2),  
			nn.LeakyReLU())  # 38,57
		# self.decoderl2 = nn.Sequential(
		# 	nn.ConvTranspose2d(16, 8, 2, stride=2),  
		# 	nn.ReLU())	 #76, 114
		self.decoderl1 = nn.Sequential(
			nn.ConvTranspose2d(hidden[4], 1, [scanx,scany]),
			nn.ReLU()
			#nn.Sigmoid()
			)
			
		self.enbedimx = int(((fgx - scanx + 1)/2-1)/2 -1)
		self.enbedimy = int(((fgy - scany + 1)/2-1)/2 -1)
		node_int = int(self.enbedimx * self.enbedimy * hidden[2])
		self.fc_mu = nn.Linear(node_int, latent_dim)
		self.fc_var = nn.Linear(node_int, latent_dim)
		self.decoder_input = nn.Linear(self.latent_dim + self.total_cluster + 1, node_int)
		
		
		if 'KLDw' in kwargs:
			self.kld_weight = kwargs['KLDw']
		else:
			self.kld_weight = 1
		
		self.seed = 0

	def encode(self, input: Tensor) -> List[Tensor]:
		"""
		Encodes the input by passing through the encoder network
		and returns the latent codes.
		:param input: (Tensor) Input tensor to encoder [N x C x H x W]
		:return: (Tensor) List of latent codes
		"""
		result = self.encoderl1(input)
		# result = self.encoderl2(result)
		result = self.encoderl3(result)
		result = self.encoderl4(result)
		result = torch.flatten(result, start_dim=1)

		# Split the result into mu and var components
		# of the latent Gaussian distribution
		mu = self.fc_mu(result)
		log_var = self.fc_var(result)

		return [mu, log_var]

	def decode(self, z: Tensor) -> Tensor:
		"""
		Maps the given latent codes
		onto the image space.
		:param z: (Tensor) [B x D]
		:return: (Tensor) [B x C x H x W]
		"""
		result = self.decoder_input(z)
		result = result.view(-1, self.hidden[2], self.enbedimx, self.enbedimy)
		result = self.decoderl4(result)
		result = self.decoderl3(result)
		# result = self.decoderl2(result)
		result = self.decoderl1(result)
		return result

	def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
		"""
		Reparameterization trick to sample from N(mu, var) from
		N(0,1).
		:param mu: (Tensor) Mean of the latent Gaussian [B x D]
		:param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
		:return: (Tensor) [B x D]
		"""
		std = torch.exp(0.5 * logvar)
		torch.manual_seed(self.seed)
		eps = torch.randn_like(std)
		return eps * std + mu

	def forward(self, input0: Tensor,input1: Tensor, **kwargs) -> List[Tensor]:
		mu, log_var = self.encode(input0)
		z = self.reparameterize(mu, log_var)
		zplus = torch.cat((z, input1), dim = 1)
		return  [self.decode(zplus), [input0,input1], mu, log_var]
	
	def loss_function(self,*args,**kwargs) -> dict:
		
		recons = args[0]
		input = args[1]
		mu = args[2]
		log_var = args[3]

		kld_weight = self.kld_weight  # Account for the minibatch samples from the dataset
		
		
		recons_loss = F.mse_loss(recons, input[0])


		kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

		loss = recons_loss + kld_weight * kld_loss
		
		return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}
		
		
class RVAE(ClusterVAE):
	def __init__(self,  
		# in_channels: int,
		latent_dim: int,
		total_cluster: int,
		hidden: List = None,
		fgx = 2, fgy = 2,
		**kwargs) -> None:
		super(RVAE, self).__init__(latent_dim, total_cluster, hidden, fgx, fgy,  **kwargs)
	
	def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
		mu, log_var = self.encode(input[0])
		z = self.reparameterize(mu, log_var)
		#print(z.shape,input[1].shape)
		zplus = torch.cat((z, input[1]), dim = 1)
		#print(zplus.shape)
		mask = (input[0] != 0) * 1
		return  [self.decode(zplus), input, mu, log_var, mask.float()]

	def loss_function(self,
					*args,
					**kwargs) -> dict:
		recons = args[0]
		input = args[1]
		mu = args[2]
		log_var = args[3]
		mask = args[4]

		kld_weight = self.kld_weight  # Account for the minibatch samples from the dataset
		
		
		recons_loss = F.mse_loss(recons * mask, input[0] * mask)


		kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

		loss = recons_loss + kld_weight * kld_loss
		return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

	def loss_ELBO(self,beta,sigma,*args,**kwargs) -> dict:
		recons = args[0]
		input = args[1]
		mu = args[2]
		log_var = args[3]
		mask = args[4]

		kld_weight = self.kld_weight  # Account for the minibatch samples from the dataset
		
		if beta > 0:
			# Calculate Gaussian CE loss
			Dim = recons.shape[1]
			const1 = -(1+beta)/beta
			const2 = 1 / pow((2 * math.pi * (sigma**2)), (beta * Dim / 2))
			SE_loss = torch.sum((recons.view(-1,recons.shape[2]*recons.shape[3]) - input[0].view(-1,input[0].shape[2]*input[0].shape[3]))**2,1)
			term1 = torch.exp(-(beta / (2 * (sigma**2))) * SE_loss)
			BBCE = torch.sum(const1*(const2* term1-1))
			#print(const1,const2,BBCE)
		else:
			BBCE = F.mse_loss(recons * mask, input[0] * mask)


		kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

		loss = BBCE + kld_weight * kld_loss
		#print(BBCE,kld_loss)
		return {'loss': loss, 'KLD':-kld_loss}
	
	def weight_reset(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
				m.reset_parameters()
		

def parse_args():
	parser = argparse.ArgumentParser(description="Train and generate using ClusterVAE on spatial transcriptomic data")
	parser.add_argument('--input_h5ad', type=str, required=True, help='Path to input .h5ad file')
	parser.add_argument('--gene_img_path', type=str, required=True, help='Path to full_geneimg.npy file')
	parser.add_argument('--cluster_path', type=str, required=True, help='Path to cluster_k_*.npy file')
	parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
	parser.add_argument('--optimal_k', type=int, default=20, help='Number of clusters (k)')
	parser.add_argument('--noise_type', type=str,default=None, help='Type of noise to apply')
	parser.add_argument('--frac_anom', type=float, default=0.05, help='Fraction of cells to contaminate')
	parser.add_argument('--beta', type=float, default=0.03, help='Beta value for ELBO loss')
	parser.add_argument('--sigma', type=float, default=0.5, help='Sigma for Gaussian')
	parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
	parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
	parser.add_argument('--nsample', type=int, default=2, help='Number of synthetic samples to generate')
	return parser.parse_args()

def main():
	args = parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	os.makedirs(args.output_dir, exist_ok=True)

    # Load data
	adata = sc.read(args.input_h5ad)
	gene_img = np.load(args.gene_img_path)
	clusters = np.load(args.cluster_path)
	
	full_dataset = cel.datagenemapclust(gene_img, clusters)
	print("Size of spatial data [observations x genes]", adata.X.shape)

    # Prepare DataLoader
	trainloader = DataLoader(full_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)

    # Initialize model
	latent_dim = 511 - clusters.max()
	CVAEmodel = RVAE(latent_dim=latent_dim, total_cluster=clusters.max(), fgx=gene_img.shape[2], fgy=gene_img.shape[3], KLDw=0, hidden=[8, 4, 2, 4, 4])
	CVAEmodel = CVAEmodel.to(device)
	optimizer = optim.Adam(CVAEmodel.parameters(), lr=1e-5, weight_decay=1e-5)

    # Training loop
	loss_min = float('inf')
	learning_rate = 1e-5
	RCcount, RCcountMax = 0, 40
	for epoch in range(args.epochs):
		total_loss = 0
		for img in tqdm(trainloader):
			img[0], img[1] = img[0].to(device), img[1].to(device)
			recon = CVAEmodel(img)
			loss = CVAEmodel.loss_ELBO(args.beta, args.sigma, *recon)
			loss['loss'].backward()
			optimizer.step()
			optimizer.zero_grad()
			total_loss += loss['loss'].item()
		
		print(f"Epoch:{epoch+1}, Loss:{total_loss/len(trainloader):.4f}")
		if total_loss > loss_min:
			RCcount += 1
			if RCcount == RCcountMax:
				RCcount = 0
				learning_rate /= 2
				optimizer.param_groups[0]['lr'] = learning_rate
				loss_min += 10
				print(f"New learning rate: {learning_rate}")
		else:
			loss_min = total_loss

    # Save model
	model_path = os.path.join(args.output_dir, f"CVAE_k_{args.optimal_k}_{args.beta}.obj")
	with open(model_path, 'wb') as f:
		pickle.dump(CVAEmodel, f)
	print(f"Saved model to: {model_path}")

    # Regenerate synthetic data
	CVAEmodel = CVAEmodel.to('cpu')
	trainloader = DataLoader(full_dataset, batch_size=1, num_workers=4, shuffle=True)
	output = []
	for img in tqdm(trainloader):
		samples = []
		mu, log_var = CVAEmodel(img)[2:4]
		for _ in range(args.nsample):
			z = CVAEmodel.reparameterize(mu, log_var)
			zplus = torch.cat((z, img[1]), dim=1)
			decoded = CVAEmodel.decode(zplus)
			samples.append(decoded.detach().numpy()[0, 0, :, :])
		output.append(np.stack(samples))
		
	data_gen = np.stack(output)
	data_gen = np.swapaxes(data_gen, 0, 1)
	data_gen = data_gen.reshape((data_gen.shape[0], data_gen.shape[1], -1))

    # Reshape to match spatial coordinates
	refer = adata.obs
	x, y = refer.iloc[:, 0], refer.iloc[:, 1]
	xmin, xmax = x.min(), x.max()
	ymin, ymax = y.min(), y.max()
	xlen, ylen = xmax - xmin + 1, ymax - ymin + 1

	marker = np.zeros(xlen * ylen, dtype=bool)
	for i in range(refer.shape[0]):
		marker[(x.iloc[i] - xmin) * ylen + y.iloc[i] - ymin] = True
		
	data_gen_rs = data_gen[:, :, marker]
	
	if not args.noise_type:
		gen_path = os.path.join(args.output_dir, f"data_gen_k_{args.optimal_k}_{args.beta}_n{args.nsample}.npy")
	else:
		gen_path = os.path.join(args.output_dir, f"data_gen_k_{args.optimal_k}_{args.noise_type}{args.frac_anom}_{args.beta}_n{args.nsample}.npy")
	np.save(gen_path, data_gen_rs)
	print(f"Saved generated data to: {gen_path}")

if __name__ == '__main__':
    main()
