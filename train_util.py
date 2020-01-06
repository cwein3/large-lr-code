
import os
import shutil
import time
import update_loss_util

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F

def load_data(dataset, batch_size, dataset_path="./", augment=True):
	normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
									 std=[x/255.0 for x in [63.0, 62.1, 66.7]])

	if augment:
		transform_train = transforms.Compose([
			transforms.ToTensor(),
			transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
								(4,4,4,4),mode='reflect').squeeze()),
			transforms.ToPILImage(),
			transforms.RandomCrop(32),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
			])
	else:
		transform_train = transforms.Compose([
			transforms.ToTensor(),
			normalize,
			])
	transform_test = transforms.Compose([
		transforms.ToTensor(),
		normalize
		])

	kwargs = {'num_workers': 1, 'pin_memory': True}
	train_loader = torch.utils.data.DataLoader(
		datasets.__dict__[dataset.upper()](dataset_path, train=True, download=True,
						 transform=transform_train),
		batch_size=batch_size, shuffle=True, **kwargs)
	val_loader = torch.utils.data.DataLoader(
		datasets.__dict__[dataset.upper()](dataset_path, train=False, transform=transform_test),
		batch_size=batch_size, shuffle=True, **kwargs)
	return train_loader, val_loader

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

def train_loop(
	train_loader,
	model,
	criterion,
	optimizer,
	epoch,
	hparams,
	print_freq=10):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()

	model.train()

	end = time.time()
	for i, (input_data, target) in enumerate(train_loader):
		target = target.cuda(async=True)
		input_data = input_data.cuda()
		input_var = torch.autograd.Variable(input_data)
		target_var = torch.autograd.Variable(target)

		loss, output, time_taken = update_loss_util.update_step(
			criterion, 
			optimizer, 
			model,
			input_var,
			target_var,
			hparams)
		prec1 = accuracy(output.data, target, topk=(1,))[0]
		losses.update(loss.data.item(), input_data.size(0))
		top1.update(prec1.item(), input_data.size(0))

		batch_time.update(time_taken)

		if i % print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					  epoch, i, len(train_loader), batch_time=batch_time,
					  loss=losses, top1=top1))

	return top1.avg, losses.avg

def validate(
	val_loader, 
	model, 
	criterion, 
	epoch,
	print_freq=10
	):
	"""Perform validation on the validation set"""
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()
	for i, (input_data, target) in enumerate(val_loader):
		target = target.cuda(async=True)
		input_data = input_data.cuda()
		input_var = torch.autograd.Variable(input_data)
		target_var = torch.autograd.Variable(target)

		# compute output
		with torch.no_grad():
			output = model(input_var)
		loss = criterion(output, target_var)

		# measure accuracy and record loss
		prec1 = accuracy(output.data, target, topk=(1,))[0]
		losses.update(loss.data.item(), input_data.size(0))
		top1.update(prec1.item(), input_data.size(0))

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % print_freq == 0:
			print('Test: [{0}/{1}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					  i, len(val_loader), batch_time=batch_time, loss=losses,
					  top1=top1))

	print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

	return top1.avg, losses.avg

def create_optimizer(model, hparams):
	return optim.SGD(
		[{'params': model.parameters(), 'lr': hparams['base_lr'], 'initial_lr': hparams['base_lr']}], 
		momentum=hparams['momentum'],
		weight_decay=hparams['weight_decay'],
		nesterov=True
		)

def adjust_lr(optimizer, epoch, lr, hparams):
	if hparams['lr_sched'] == 'default':
		sched_func = lr_sched_default
	if hparams['lr_sched'] == 'fixed':
		sched_func = lr_fixed
	if hparams['lr_sched'] == 'decay_once':
		sched_func = lr_decay_once
	if hparams['lr_sched'] == 'decay_early':
		sched_func = lr_decay_early
	lr_vals = []
	for param_group in optimizer.param_groups:
		param_group['lr'] = sched_func(param_group['initial_lr'], epoch)
		lr_vals.append(param_group['lr'])
	return lr_vals

def lr_sched_default(lr, epoch):
	new_lr = lr*(0.2**int(epoch >= 60))
	new_lr *= (0.2**int(epoch >= 120))
	new_lr *= (0.2**int(epoch >= 150))
	return new_lr

def lr_fixed(lr, epoch):
	return lr

def lr_decay_once(lr, epoch):
	new_lr = lr*(0.2**int(epoch >= 150))
	return new_lr

def lr_decay_early(lr, epoch):
	new_lr = lr*(0.04**int(epoch >= 30))
	return new_lr

def adjust_act_noise(decay_type, decay_rate, noise, epoch):
	if decay_type == "none":
		return noise
	if decay_type == "step":
		return act_noise_decay_step(noise, epoch, decay_rate)
	if decay_type == 'step_early':
		return act_noise_decay_step_early(noise, epoch, decay_rate)
	if decay_type == "cont":
		return act_noise_decay_cont(noise, epoch, decay_rate)

def act_noise_decay_step(noise, epoch,decay_rate):
	new_noise = noise*(decay_rate**int(epoch >= 60))
	new_noise *= (decay_rate**int(epoch >= 120))
	new_noise *= (decay_rate**int(epoch >= 150))
	return new_noise

def act_noise_decay_step_early(noise, epoch, decay_rate):
	new_noise = noise*(decay_rate**int(epoch >= 30))
	return new_noise

def act_noise_decay_cont(noise, epoch, decay_rate):
	return noise*(decay_rate**(epoch - 1))

