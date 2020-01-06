
import os
import numpy as np 
import update_loss_util
import matplotlib.pyplot as plt 
import json
import torch
import shutil

def write_args(args, save_dir):
	args = vars(args)
	args_file = os.path.join(save_dir, "args.txt")
	with open(args_file, "w") as args_f:
		args_f.write(str(args))

def log_scalar_file(scalar_dict, epoch, filename):
	with open(filename, "a") as f:
		f.write("Epoch %d, Data %s\n" % (epoch, json.dumps(scalar_dict)))

def load_scalar_dict(save_dir):
	data_loc= os.path.join(save_dir, 'scalar_dict.pkl')
	return torch.load(data_loc)

def make_scalar_plots(
	scalar_dict, 
	save_dir):
	keylist = scalar_dict[1].keys()
	num_epochs = len(scalar_dict)
	for key in keylist:
		epochs = range(1, num_epochs + 1)
		vals = [scalar_dict[epoch][key] for epoch in epochs]
		plt.plot(
			epochs,
			vals,
			label=key)
		save_name= os.path.join(save_dir, key + "_time")
		plt.legend()
		plt.savefig(save_name)
		plt.close()
		
def save_checkpoint(
	model_state,
	data_state,
	is_best,
	save_dir):
	
	filename = os.path.join(save_dir, 'checkpoint.pth.tar')
	torch.save(model_state, filename)
	if is_best:
		shutil.copyfile(filename, os.path.join(save_dir, 'model_best.pth.tar'))

	data_loc = os.path.join(save_dir, 'scalar_dict.pkl')
	torch.save(data_state, data_loc)