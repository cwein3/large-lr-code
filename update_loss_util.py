import math
import torch
import subprocess
import itertools
import numpy as np 
import torch.nn as nn
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt 

def update_step(
	criterion,
	optimizer,
	model,
	inputs,
	labels,
	hparams):
	if hparams["reg_type"] in ["none", "dropout"]:
		return none_update(
			criterion,
			optimizer,
			model,
			inputs,
			labels)
	if hparams["reg_type"] == "act_noise":
		return noise_update(
			criterion,
			optimizer,
			model,
			inputs,
			labels,
			hparams)

def none_update(
	criterion,
	optimizer,
	model,
	inputs,
	labels):
	start_time = time.time()
	output = model(inputs)
	loss = criterion(output, labels)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	return loss, output, time.time() - start_time

def noise_update(
	criterion,
	optimizer,
	model,
	inputs,
	labels,
	hparams):
	start_time = time.time()
	output = model.forward_noise(inputs, hparams["noise_level"])
	loss = criterion(output, labels)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	return loss, output, time.time() - start_time