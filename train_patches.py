import argparse as ap
import os
import shutil
import copy
from datetime import datetime

import torch 
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn

import save_util
import train_util
import update_loss_util
import data_util

import models
import pickle
import numpy as np 

def train_model(
	model_specs,
	save_dir,
	train_loader,
	val_clean_loader,
	val_aug_loader,
	transform_only_loader):
	print("***************************************************************************")
	print("Training model:", model_specs["name"])
	model_args = {
		"num_classes": 10
	}
	model = models.__dict__['wideresnet16'](**model_args)

	print('Number of model parameters: {}'.format(
		sum([p.data.nelement() for p in model.parameters()])))

	model = model.cuda()

	cudnn.benchmark = True

	criterion = nn.CrossEntropyLoss().cuda()
	optim_hparams = {
		'base_lr' : model_specs['lr'], 
		'momentum' : 0.9,
		'weight_decay' : 5e-4,
		'lr_type' : 'default'
	}
	lr_hparams = {'lr_sched' : model_specs['lr_sched']}
	optimizer = train_util.create_optimizer(
		model,
		optim_hparams)

	save_dir = os.path.join(save_dir, model_specs["name"])
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	scalar_summary_file = os.path.join(save_dir, "scalars.txt")
	scalar_dict = {}
	best_val = 0

	for epoch in range(model_specs['epochs']):
		lr = train_util.adjust_lr(
			optimizer,
			epoch + 1,
			model_specs['lr'],
			lr_hparams)

		train_hparams = {
			"reg_type" : model_specs['reg_type'],
			"noise_level" : train_util.adjust_act_noise(
				model_specs['act_noise_decay'],
				model_specs['act_noise_decay_rate'],
				model_specs['act_noise'],
				epoch + 1)
		}

		train_acc, train_loss = train_util.train_loop(
			train_loader,
			model,
			criterion,
			optimizer,
			epoch,
			train_hparams,
			print_freq=10)

		print("Validating clean accuracy.")
		val_clean_acc, val_clean_loss = train_util.validate(
			val_clean_loader,
			model,
			criterion,
			epoch,
			print_freq=10)

		print("Validating mixed accuracy")
		val_aug_acc, val_aug_loss = train_util.validate(
			val_aug_loader,
			model,
			criterion,
			epoch,
			print_freq=10)

		print("Validating additional transforms only.")
		t_acc, t_loss = train_util.validate(
			transform_loader,
			model,
			criterion,
			epoch,
			print_freq=2)

		scalar_epoch = {
			"lr": lr, 
			"train_loss": train_loss, 
			"train_acc": train_acc, 
			"val_clean_loss": val_clean_loss, 
			"val_clean_acc": val_clean_acc,
			"val_aug_loss": val_aug_loss,
			"val_aug_acc": val_aug_acc,
			"patch_acc": t_acc
		}

		scalar_dict[epoch + 1] = scalar_epoch

		save_util.log_scalar_file(
			scalar_epoch,
			epoch + 1,
			scalar_summary_file)

		save_util.make_scalar_plots(
			scalar_dict,
			save_dir
		)

		is_best = val_aug_acc > best_val
		best_val = max(val_aug_acc, best_val)

		save_util.save_checkpoint(
			{
				'epoch': epoch + 1,
				'state_dict': model.state_dict(),
				'best_val': best_val,
			}, 
			scalar_dict,
			is_best,
			save_dir)

		print('Best accuracy: ', best_val)
		
model_names = sorted(name for name in models.__dict__
	if name.islower() and not name.startswith("__")
	and callable(models.__dict__[name]))

parser = ap.ArgumentParser(description='Synthetic patch experiments.')
parser.add_argument('--epochs', default=60, type=int, 
					help='How many epochs to train.')
parser.add_argument('--batch_size', default=128, type=int, 
					help='Batch size.')
parser.add_argument('--print_freq', default=10, type=int,
					help='How frequently to log iterations.')
parser.add_argument('--save_dir', type=str,
					help='Where to save the experimental runs.')
parser.add_argument('--data_dir', type=str,
					help='Where the CIFAR data is stored.')

# Patch arguments
parser.add_argument('--patch_only_scale', type=float, default=1.75,
					help='Scaling for patch only imgs.')
parser.add_argument('--patch_sep', type=float, default=0.1,
					help='Patch is generated as z + some random pattern with coordinates in [-patch_sep, patch_sep].')
parser.add_argument('--patch_size', type=int, default=7,
					help='Size of the patch in pixels.')
parser.add_argument('--num_patterns', type=int, default=1, 
					help='How many patch patterns to have.')
parser.add_argument('--patch_only_prob', type=float, default=0.2,
					help='Probability of an image having only a patch, if it is chosen as augmented iamge.')
parser.add_argument('--z_scale', type=float, default=1.25,
					help='Size of the z vector.')
parser.add_argument('--seed', type=int, default=2, help='Random seed for experiments.')


args=parser.parse_args()
print(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

timestamp = datetime.utcnow().strftime("%H_%M_%S_%f-%d_%m_%y")
save_str = "patch_%s" % (
		timestamp)

save_dir = os.path.join(args.save_dir, save_str)
patch_hparams = {
	'transforms' : 'pm_sym_fixed',
	'num_classes' : 10,
	'patch_size' : args.patch_size,
	'num_patterns' : args.num_patterns,
	'patch_sep' : args.patch_sep,
	'z_scale' : args.z_scale,
	'patch_loc' : 'center',
	'patch_only_prob' : args.patch_only_prob,
	'patch_only_scale' : args.patch_only_scale
}

m_trans = data_util.create_m_trans(patch_hparams)

loader_hparams = {
	'batch_size' : args.batch_size,
	'prop_augment' : 0.8,
	'dataset' : 'cifar10',
	'data_dir' : args.data_dir	
}

train_loader, val_clean_loader, val_aug_loader = data_util.load_data(
	loader_hparams,
	augment=False,
	more_transform=m_trans)

# HACKY
m_trans_copy = [copy.copy(m_t) for m_t in m_trans]
m_trans_copy[0].sample_patch = False
transform_loader = data_util.transform_only_loader(
	128,
	length=1000,
	n_classes=10,
	m_transform=m_trans_copy)
data_util.vis_imgs('cifar10', args.data_dir, m_trans_copy, save_dir)
# HACKY


large_lr_spec = {
	'name' : 'large_lr', 
	'lr' : 0.1, 
	'reg_type': 'none',
	'lr_sched': 'decay_early',
	'act_noise' : None,
	'act_noise_decay' : None,
	'act_noise_decay_rate' : None,
	'epochs' : args.epochs
}

small_lr_spec = {
	'name' : 'small_lr',
	'lr' : 0.004,
	'reg_type' : 'none',
	'lr_sched' : 'fixed',
	'act_noise' : None,
	'act_noise_decay' : None,
	'act_noise_decay_rate' : None,
	'epochs' : args.epochs
}

act_noise_spec = {
	'name' : 'act_noise', 
	'lr' : 0.004, 
	'reg_type' : 'act_noise', 
	'lr_sched' : 'fixed',
	'act_noise' : 4e-1,
	'act_noise_decay' : 'step_early',
	'act_noise_decay_rate' : 1e-5,
	'epochs' : args.epochs
}

model_specs = [large_lr_spec, small_lr_spec, act_noise_spec]
for spec in model_specs:
	train_model(
		spec,
		save_dir,
		train_loader,
		val_clean_loader,
		val_aug_loader,
		transform_loader)


