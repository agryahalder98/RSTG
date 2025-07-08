# dnn_layer_prediction.py
import argparse
import torch
import numpy as np
import pickle
import scanpy as sc
import celery as cel
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import tensor as Tensor
import os

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom dataset
class wrap_gene_layer(TensorDataset):
    def __init__(self, datainput, label, layerkey="Layer"):
        self.data_tensor = torch.from_numpy(datainput).float()
        getlayer = label[layerkey].to_numpy()
        self.layer = getlayer.astype('float32')
        self.layersunq = np.sort(np.unique(self.layer))
        self.nlayers = len(self.layersunq)
        self.imagedimension = self.data_tensor.shape

    def __getitem__(self, index):
        indexsample = index // self.imagedimension[2]
        indexspot = index % self.imagedimension[2]
        geneseq = self.data_tensor[indexsample,:,indexspot]
        layeri = int(self.layer[indexspot]) - 1
        layerv = np.zeros(self.nlayers-1)
        layerv[:layeri] = 1
        return geneseq, layerv

    def __len__(self):
        return self.imagedimension[0] * self.imagedimension[2]

class DNN(nn.Module):

	def __init__(self,  
				 in_channels: int,
				 hidden_dims: List = None,			 
				 **kwargs) -> None:
		super(DNN, self).__init__()
		
		if hidden_dims is None:
			hidden_dims = [200, 100, 50]

		self.fclayer1 = nn.Sequential( 
			nn.Linear(in_channels, hidden_dims[0]),
			# nn.BatchNorm1d(hidden_dims[0]),
			nn.ReLU())
		self.fclayer2 = nn.Sequential( 
			nn.Linear(hidden_dims[0], hidden_dims[1]),
			# nn.BatchNorm1d(hidden_dims[1]),
			nn.ReLU())
		self.fclayer3 = nn.Sequential( 
			nn.Linear(hidden_dims[1], hidden_dims[2]),
			# nn.BatchNorm1d(hidden_dims[2]),
			nn.ReLU())
		self.fclayer4 = nn.Sequential(
			nn.Linear(hidden_dims[2], 2),
			# nn.BatchNorm1d(2),
			nn.Sigmoid())

	def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
		z = self.fclayer1(input[0])
		z = self.fclayer2(z)
		z = self.fclayer3(z)
		z = self.fclayer4(z)
		return  [z,input]

	def loss_function(self,
					  *args,
					  **kwargs) -> dict:
		"""
		Computes the spatial coordinates loss function
		:param args: results data and input matrix
		:return:
		"""
		cord_pred = args[0]
		input = args[1]

		loss = F.mse_loss(cord_pred, input[1])
		
		return {'loss': loss}


class DNNordinal(DNN):
	def __init__(self,  
		# in_channels: int,
		in_channels: int,
		num_classes: int,
		hidden_dims: List = None,	
		importance_weights: List=None,
		**kwargs) -> None:
		super(DNNordinal, self).__init__(in_channels, hidden_dims, **kwargs)
		
		if hidden_dims is None:
			hidden_dims = [200, 100, 50]

		self.fclayer1 = nn.Sequential( 
			nn.Linear(in_channels, hidden_dims[0]),
			nn.Dropout(0.25),
			nn.ReLU())
		self.fclayer2 = nn.Sequential( 
			nn.Linear(hidden_dims[0], hidden_dims[1]),
			nn.ReLU())
		self.fclayer3 = nn.Sequential( 
			nn.Linear(hidden_dims[1], hidden_dims[2]),
			nn.ReLU())
		self.fclayer4 = nn.Sequential(
			nn.Linear(hidden_dims[2], 1))
		
		self.coral_bias = torch.nn.Parameter(
                torch.arange(num_classes-1, 0, -1).float() / (num_classes-1))
				
		self.importance_weights = importance_weights
	
	def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
		"""
        Computes forward pass.
        Parameters
        -----------
        x : torch.tensor, shape=(num_examples, num_features)
            Input features.
        Returns
        -----------
        logits : torch.tensor, shape=(num_examples, num_classes-1)
        """
		z = self.fclayer1(input[0])
		z = self.fclayer2(z)
		z = self.fclayer3(z)
		z = self.fclayer4(z)
		logits = z + self.coral_bias
		#print(logits.shape)
		return  [logits, input]

	def loss_function(self,device,
					*args,
					**kwargs) -> dict:
		"""Computes the CORAL loss described in
		Cao, Mirjalili, and Raschka (2020)
		*Rank Consistent Ordinal Regression for Neural Networks
		   with Application to Age Estimation*
		Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008
		Parameters
		----------
		logits : torch.tensor, shape(num_examples, num_classes-1)
			Outputs of the CORAL layer.
		levels : torch.tensor, shape(num_examples, num_classes-1)
			True labels represented as extended binary vectors
			(via `coral_pytorch.dataset.levels_from_labelbatch`).
		importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
			Optional weights for the different labels in levels.
			A tensor of ones, i.e.,
			`torch.ones(num_classes, dtype=torch.float32)`
			will result in uniform weights that have the same effect as None.
		reduction : str or None (default='mean')
			If 'mean' or 'sum', returns the averaged or summed loss value across
			all data points (rows) in logits. If None, returns a vector of
			shape (num_examples,)

		"""
		logits = args[0]
		levels = args[1][1]
		#print(logits.shape, levels.shape)
		
		if not logits.shape == levels.shape:
			raise ValueError("Please ensure that logits (%s) has the same shape as levels (%s). "
							% (logits.shape, levels.shape))
		term1 = (F.logsigmoid(logits)*levels  + (F.logsigmoid(logits) - logits)*(1-levels))
		layerid = torch.sum(levels, dim = 1).to(torch.int)
		
		if self.importance_weights is not None:
			self.importance_weights = self.importance_weights.to(device)
			term2 =  torch.mul(self.importance_weights[layerid].to(device), term1.transpose(0,1))
		else:
			term2 =  term1.transpose(0,1)
		
		val = (-torch.sum(term2, dim=0))

		loss = torch.mean(val)
		return {'loss': loss}

def main(args):
    data = sc.read(args.input_h5ad)

    os.makedirs(args.output_dir, exist_ok=True)

    cel.get_zscore(data)
    data = data[data.obs.sort_values(by=['x2', 'x3']).index]
    drop_loc = np.where(data.obs["Layer"]==0)[0]
    data = data[data.obs["Layer"]!=0]

    data_gen_rs = np.load(args.input_npy)#.format(beta=args.beta, nrep=args.nrep))
    tdatax = np.expand_dims(data.X, axis=0)
    tdata_rs = np.swapaxes(tdatax, 1, 2)
    datacomp = np.concatenate((data_gen_rs, tdata_rs), axis=0)

    dataset = wrap_gene_layer(datacomp, data.obs, "Layer")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    layer_count =  data.obs["Layer"].value_counts().sort_index()
    layer_weight = layer_count[7]/layer_count[0:7]
    layer_weights = torch.tensor(layer_weight.to_numpy())
    

    model = DNNordinal( in_channels = data_gen_rs.shape[1], num_classes = 7, hidden_dims = [200, 100, 50], importance_weights = layer_weights )
    model = model.float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_min = float('inf')
    RCcount, RCcountMax = 0, 5
    learning_rate = 1e-3
    
    for epoch in range(args.epochs):
        total_loss = 0
        for img in tqdm(dataloader):
            img[0], img[1] = img[0].to(device), img[1].to(device)
            recon = model(img)
            loss = model.loss_function(device,*recon)
            loss.get('loss').backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.get('loss').item()

        print(f"Epoch: {epoch+1}, Loss: {total_loss:.4f}")

        if total_loss > loss_min:
            RCcount += 1
            if RCcount == RCcountMax:
                RCcount = 0
                learning_rate /= 2
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
                loss_min += 10
                print(f"New learning rate: {learning_rate:.6f}")
        else:
            loss_min = total_loss

        if learning_rate < 1e-7:
            break

    filename = os.path.splitext(os.path.basename(args.input_npy))[0] + '.obj'
    save_path = os.path.join(args.output_dir,filename )
    print(save_path)
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DNN model for spatial transcriptomics layer prediction")
    parser.add_argument('--input_h5ad', type=str, required=True, help='Path to input .h5ad file')
    parser.add_argument('--input_npy', type=str, required=True, help='Path to generated data .npy file')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save trained model')
    parser.add_argument('--beta', type=float, default=0.005, help='Beta divergence value')
    parser.add_argument('--nrep', type=int, default=2, help='Repetition number')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')

    args = parser.parse_args()
    main(args)
