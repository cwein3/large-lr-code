import argparse
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

import models
import pickle

model_names = sorted(name for name in models.__dict__
	if name.islower() and not name.startswith("__")
	and callable(models.__dict__[name]))
reg_types = ["none", "act_noise", "dropout"]
lr_scheds = ["default", "fixed", "decay_once","decay_early"]
act_noise_decay = ["none", "step", "cont", "step_early"]

parser = argparse.ArgumentParser(description='Mitigation strategies for a small learning rate.')
parser.add_argument('--epochs', default=200, type=int,
					help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int,
					help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=128, type=int,
					help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
					help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
					help='weight decay (default: 5e-4)')
parser.add_argument('--print_freq', '-p', default=10, type=int,
					help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str,
					help='path to latest checkpoint (default: none)')
parser.add_argument('--reg_type', choices=reg_types, default='none',
					help=' | '.join(reg_types))
parser.add_argument('--dropout', type=float, default=0.4, help='amount of dropout to have.')
parser.add_argument('--dataset', choices=["cifar10", "cifar100"], default="cifar10",
					help='cifar10 or cifar100')
parser.add_argument('--lr_sched', choices=lr_scheds, default='default', 
					help=' | '.join(lr_scheds))
parser.add_argument('--arch', choices=model_names, default="wideresnet16",
					help='model architecture:' + ' | '.join(model_names))
parser.add_argument('--act_noise_decay', choices=act_noise_decay, default='none',
					help='Type of decay on activation noise.')
parser.add_argument('--act_noise_decay_rate', type=float, default=0.995, help='Decay rate of activaiton noise.')
parser.add_argument('--act_noise', type=float, default=1e-2, help='Level of activation noise to add.')
parser.add_argument('--no_augment', action='store_true', 
					help='whether to have data augmentation')
parser.add_argument('--save_dir', type=str, help='location to save all the experimental runs.')
parser.add_argument('--data_dir', type=str, help='where the CIFAR data is located.')
parser.set_defaults(no_augment=False)

def main():
	args = parser.parse_args()
	for arg in vars(args):
		print(arg, " : ", getattr(args, arg))
	timestamp = datetime.utcnow().strftime("%H_%M_%S_%f-%d_%m_%y")
	save_str = "arch_%s_reg_%s_%s" % (
		args.arch,
		args.reg_type,
		timestamp)
	save_dir = os.path.join(args.save_dir, save_str)

	augment = not args.no_augment
	train_loader, val_loader = train_util.load_data(
		args.dataset, 
		args.batch_size, 
		dataset_path=args.data_dir,
		augment=augment)
	
	print("=> creating model '{}'".format(args.arch))
	model_args = {
		"num_classes": 10 if args.dataset == "cifar10" else 100	
	}
	if args.reg_type == 'dropout':
		print("Using dropout.")
		model_args['dropRate'] = args.dropout
	model = models.__dict__[args.arch](**model_args)

	print('Number of model parameters: {}'.format(
		sum([p.data.nelement() for p in model.parameters()])))

	model = model.cuda()

	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			best_val = checkpoint['best_val']
			model.load_state_dict(checkpoint['state_dict'])
			print("=> loaded checkpoint '{}' (epoch {})"
				.format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	cudnn.benchmark = True

	criterion = nn.CrossEntropyLoss().cuda()
	optim_hparams = {
		'base_lr' : args.lr, 
		'momentum' : args.momentum,
		'weight_decay' : args.weight_decay
	}
	lr_hparams = {'lr_sched' : args.lr_sched}
	optimizer = train_util.create_optimizer(
		model,
		optim_hparams)

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	save_util.write_args(args, save_dir)
	scalar_summary_file = os.path.join(save_dir, "scalars.txt")
	scalar_dict = {}
	best_val = 0
	all_dict = {}

	for epoch in range(args.start_epoch, args.epochs):
		lr = train_util.adjust_lr(
			optimizer,
			epoch + 1,
			args.lr,
			lr_hparams)

		train_hparams = {
			"reg_type" : args.reg_type,
			"noise_level" : train_util.adjust_act_noise(
				args.act_noise_decay,
				args.act_noise_decay_rate,
				args.act_noise,
				epoch + 1)
		}

		train_acc, train_loss = train_util.train_loop(
			train_loader,
			model,
			criterion,
			optimizer,
			epoch,
			train_hparams,
			print_freq=args.print_freq)

		print("Validating accuracy.")
		val_acc, val_loss = train_util.validate(
			val_loader,
			model,
			criterion,
			epoch,
			print_freq=args.print_freq)

		scalar_epoch = {
			"lr": lr, 
			"train_loss": train_loss, 
			"train_acc": train_acc, 
			"val_loss": val_loss, 
			"val_acc": val_acc
		}

		scalar_dict[epoch + 1] = scalar_epoch

		save_util.log_scalar_file(
			scalar_epoch,
			epoch + 1,
			scalar_summary_file)

		is_best = val_acc > best_val
		best_val = max(val_acc, best_val)

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
			
main()
