from __future__ import print_function
import argparse
import torch
import math
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from sklearn.model_selection import train_test_split
#from keras.datasets import fashion_mnist
import scipy.io as spio
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import scanpy as sc
from sklearn.cluster import KMeans
import celery as cel
import random
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F

from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import tensor as Tensor
import pickle
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import json
from scipy.sparse import issparse

import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
                torch.arange(num_classes, 0, -1).float() / (num_classes))
		#print(self.coral_bias.shape)
				
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
		#print(input[0].shape, logits.shape)
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
		#print(logits, levels)
		
		if not logits.shape == levels.shape:
			raise ValueError("Please ensure that logits (%s) has the same shape as levels (%s). "
							% (logits.shape, levels.shape))
		term1 = (F.logsigmoid(logits)*levels  + (F.logsigmoid(logits) - logits)*(1-levels))
		layerid = torch.sum(levels, dim = 1).to(torch.int)
		
		if self.importance_weights is not None:
			#self.importance_weights = self.importance_weights.to('cpu')
			#print(layerid)
			self.importance_weights = self.importance_weights.to(device)
			#print(self.importance_weights,term1)
			term2 =  torch.mul(self.importance_weights[layerid].to(device), term1.transpose(0,1))
			
		else:
			term2 =  term1.transpose(0,1)
		
		val = (-torch.sum(term2, dim=0))

		loss = torch.mean(val)
		return {'loss': loss}

def report_prop_method_LIBD (folder, tissueID, name, dataSection2, traindata, Val_loader, coloruse, class_num=7, outname = ""):
    """
        Report the results of the proposed methods in comparison to the other method
        :folder: string: specified the folder that keep the proposed DNN method
        :name: string: specified the name of the DNN method, also will be used to name the output files
        :dataSection2: AnnData: the data of Section 2
        :traindata: AnnData: the data used in training data. This is only needed for compute SSIM
        :Val_loader: Dataload: the validation data from dataloader
        :outname: string: specified the name of the output, default is the same as the name
        :ImageSec2: Numpy: the image data that are refering to
    """
    if outname == "":
        outname = name
    filename2 = r"{folder}\{name}.obj".format(folder = folder, name = name)
    #print(filename2)
    filehandler = open(filename2, 'rb') 
    #print(filehandler)
    DNNmodel = pickle.load(filehandler)
    DNNmodel = DNNmodel.to('cpu')
    #
    coords_predict = np.zeros(dataSection2.obs.shape[0])
    payer_prob = np.zeros((dataSection2.obs.shape[0],class_num+2))
    for i, img in enumerate(Val_loader):
        recon = DNNmodel(img)
        logitsvalue = np.squeeze(torch.sigmoid(recon[0]).detach().numpy(), axis = 0)
        if (logitsvalue[class_num-2] == 1):
            coords_predict[i] = class_num
            payer_prob[i,(class_num + 1)] = 1
        else:
            logitsvalue_min = np.insert(logitsvalue, 0, 1, axis=0)
            logitsvalue_max = np.insert(logitsvalue_min, class_num, 0, axis=0) 
            prb = np.diff(logitsvalue_max)
            # prbfull = np.insert(-prb[0], 0, 1 -logitsvalue[0,0], axis=0)
            prbfull = -prb.copy() 
            coords_predict[i] = np.where(prbfull == prbfull.max())[0].max() + 1
            #print(payer_prob[i,2:].shape,prbfull.shape)
            payer_prob[i,1:] = prbfull
    #
    #print(coords_predict.shape)
    dataSection2.obs["pred_layer"] = coords_predict.astype(int)
    payer_prob[:,0] = dataSection2.obs["Layer"]
    payer_prob[:,1] = dataSection2.obs["pred_layer"]
    dataSection2.obs["pred_layer_str"] = coords_predict.astype(int).astype('str')
    plot_layer(adata = dataSection2, folder = r"{folder}{tissueID}".format(folder = folder, tissueID = tissueID), name = name, coloruse = coloruse)
    plot_confusion_matrix ( referadata = dataSection2, filename = r"{folder}{tissueID}\{name}conf_mat_fig".format(folder = folder, tissueID = tissueID, name = name))
    np.savetxt(r"{folder}{tissueID}\{name}_probmat.csv".format(folder = folder, tissueID = tissueID, name = name), payer_prob, delimiter=',')



def plot_layer(adata, folder, name, coloruse=None):
    """
    This function creates and saves two scatter plots of the input AnnData object. One plot displays the predicted
    layers and the other shows the reference layers.

    :param adata: AnnData object containing the data matrix and necessary metadata
    :param folder: Path to the folder where the plots will be saved
    :param name: Prefix for the output plot file names
    :param coloruse: (Optional) List of colors to be used for the plots, default is None

    :return: None, saves two scatter plots as PDF files in the specified folder
    """

    # Define the default color palette if none is provided
    if coloruse is None:
        colors_use = ['#46327e', '#365c8d', '#277f8e', '#1fa187', '#4ac16d', '#a0da39', '#fde725', '#ffbb78', '#2ca02c',
                      '#ff7f0e', '#1f77b4', '#800080', '#959595', '#ffff00', '#014d01', '#0000ff', '#ff0000', '#000000']
    else:
        colors_use = coloruse

    if not os.path.exists(folder):
        os.mkdir(folder)
    # Define the number of cell types
    num_celltype = 7
    # Set the colors for the predicted layer in the AnnData object
    adata.uns["pred_layer_str_colors"] = list(colors_use[:num_celltype])
    # Create a copy of the input AnnData object to avoid modifying the original data
    cdata = adata.copy()
	# Scale the x2 and x3 columns in the AnnData object's observation data
    cdata.obs["x4"] = cdata.obs["x2"] * 50
    cdata.obs["x5"] = cdata.obs["x3"] * 50
    # Create and customize the predicted layer scatter plot
    fig = sc.pl.scatter(cdata, alpha=1, x="x5", y="x4", color="pred_layer_str", palette=colors_use, show=False, size=50)
    fig.set_aspect('equal', 'box')
    # Save the predicted layer scatter plot as a PDF file
    fig.figure.savefig(r"{path}\{name}_Layer_pred.pdf".format(path=folder, name=name), dpi=300)
    # Convert the 'Layer' column in the AnnData object's observation data to integer and then to string
    cdata.obs["Layer"] = cdata.obs["Layer"].astype(int).astype('str')
    # Create and customize the reference layer scatter plot
    fig2 = sc.pl.scatter(cdata, alpha=1, x="x5", y="x4", color="Layer", palette=colors_use, show=False, size=50)
    fig2.set_aspect('equal', 'box')
    # Save the reference layer scatter plot as a PDF file
    fig2.figure.savefig(r"{path}\{name}_Layer_ref.pdf".format(path=folder, name=name), dpi=300)


def plot_confusion_matrix (referadata, filename, nlayer = 7):
	""" Plot the confusion matrix
		:referadata: the main adata that are working with
		:filename: Numpy [n x 2]: the predicted coordinates based on deep neural network
	"""
	labellist = [i+1 for  i in range(nlayer)]
	conf_mat = confusion_matrix(referadata.obs[["Layer"]], referadata.obs[["pred_layer"]], labels = labellist)
	conf_mat_perc = conf_mat / conf_mat.sum(axis=1, keepdims=True)   # transform the matrix to be row percentage
	conf_mat_CR = classification_report(referadata.obs[["Layer"]], referadata.obs[["pred_layer"]], output_dict=True, labels = labellist)
	np.savetxt('{filename}.csv'.format(filename = filename), conf_mat_perc, delimiter=',')
	with open('{filename}_Classification_Metric.json'.format(filename = filename), 'w') as fp:
		json.dump(conf_mat_CR, fp)
	# plt.figure()
	# conf_mat_fig = seaheatmap(conf_mat_perc, annot=True, cmap='Blues')
	# confplot = conf_mat_fig.get_figure()    
	# confplot.savefig("{filename}.png".format(filename = filename), dpi=400)

