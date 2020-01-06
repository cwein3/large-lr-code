from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import torch
import os
import numpy as np
import torch.nn.functional as F

from PIL import Image

def load_data(hparams, augment=True, more_transform=None):
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
			normalize
			])
	else:
		transform_train = transforms.Compose([
			transforms.ToTensor(),
			normalize
			])
	if more_transform is not None:
		m_trans = transforms.Compose([t for t in more_transform])
	else:
		m_trans = None
	transform_test = transforms.Compose([
		transforms.ToTensor(),
		normalize
		])

	kwargs = {'num_workers': 1, 'pin_memory': True}
	train_loader = torch.utils.data.DataLoader(
		FixedAugmentDataset(
			hparams["dataset"],
			hparams["data_dir"],
			hparams["prop_augment"],
			train=True, 
			transforms=transform_train,
			more_transforms=m_trans),
		batch_size=hparams["batch_size"], shuffle=True, **kwargs)
	val_augment_loader = torch.utils.data.DataLoader(
		FixedAugmentDataset(
			hparams["dataset"],
			hparams["data_dir"],
			hparams["prop_augment"],
			train=False, 
			transforms=transform_test,
			more_transforms=m_trans),
		batch_size=hparams["batch_size"], shuffle=True, **kwargs)

	val_clean_loader = torch.utils.data.DataLoader(
		datasets.__dict__[hparams["dataset"].upper()](hparams["data_dir"], train=False, transform=transform_test),
		batch_size=hparams["batch_size"], shuffle=True, **kwargs)

	return train_loader, val_clean_loader, val_augment_loader

def vis_imgs(dataset, data_dir, m_transforms, save_dir):
	normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
									 std=[x/255.0 for x in [63.0, 62.1, 66.7]])

	def unnormalize(img):
		img=img.numpy()
		mean=np.array([x/255.0 for x in [125.3, 123.0, 113.9]]).reshape((3,1,1))
		std=np.array([x/255.0 for x in [63.0, 62.1, 66.7]]).reshape((3,1,1))
		img= img*std + mean
		img = img.transpose((1, 2,0))
		return img*255


	transform_train = transforms.Compose([
			transforms.ToTensor(),
			normalize
			])

	m_trans = transforms.Compose([t for t in m_transforms])
	aug_dataset = AugmentDataset(
		dataset,
		data_dir, 
		train=True,
		transforms=transform_train,
		more_transforms=m_trans)
	
	img_folder = os.path.join(save_dir, "augment_vis")
	if not os.path.exists(img_folder):
		os.makedirs(img_folder)
	img_indices = np.random.randint(50000, size=10)
	for ind in img_indices:
		img, lbl = aug_dataset[ind]
		pil_img= Image.fromarray(unnormalize(img).astype(np.uint8))
		pil_img.save(os.path.join(img_folder, "%d.png" % (ind)))

# wrap DataLoader where we can pass in arbitrary augmentations
class AugmentDataset(torch.utils.data.Dataset):
	def __init__(self, dataset, data_dir, train=True, transforms=None, more_transforms=None):
		self.dataset = datasets.__dict__[dataset.upper()](
			data_dir, 
			train=train, 
			download=True,
			transform=transforms)

		self.more_transforms = more_transforms

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		img, label = self.dataset[idx]
		if self.more_transforms is not None:
			img, label = self.more_transforms((img, label))
		return img, label

# indices of the additional augmentations are fixed
class FixedAugmentDataset(torch.utils.data.Dataset):
	def __init__(self, dataset, data_dir, augment_prob, train=True, transforms=None, more_transforms=None):
		self.dataset = datasets.__dict__[dataset.upper()](
			data_dir, 
			train=train, 
			download=True,
			transform=transforms)

		self.more_transforms = more_transforms
		self.should_transform = np.random.random((len(self.dataset),))
		self.aug_prob = augment_prob

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		img, label = self.dataset[idx]
		if self.should_transform[idx] < self.aug_prob and self.more_transforms is not None:
			img, label = self.more_transforms((img, label))			
		return img, label

def transform_only_loader(batch_size, length=1000, n_classes=10,m_transform=None):
	kwargs = {'num_workers': 1, 'pin_memory': True}
	assert(m_transform is not None) 
	m_trans = transforms.Compose([t for t in m_transform])
	loader = torch.utils.data.DataLoader(
		TransformOnlyDataset(
			length,
			n_classes,
			m_trans),
		batch_size=batch_size, shuffle=True, **kwargs)
	return loader

# apply transforms only to 0 dataset
class TransformOnlyDataset(torch.utils.data.Dataset):
	def __init__(self, length, n_classes=10, m_transform=None):
		self.len = length
		self.transforms = m_transform
		self.n_classes = n_classes

	def __len__(self):
		return self.len

	def __getitem__(self, idx):
		img = torch.zeros((3,32,32))
		label = np.random.randint(self.n_classes)
		return self.transforms((img, label))

def get_start_for_patch_loc(patch_loc, patch_size):
	if patch_loc == "corner":
		return 0
	if patch_loc == "center":
		return int(16 - patch_size/2)

# with some probability we zero out the entire image
class PatchAugmentFixed(object):
	def __init__(self, 
		z_pattern, 
		small_patterns, 
		patch_loc, 
		patch_only_prob=0,
		patch_only_scale=0.1):
		self.num_classes = small_patterns.size(0)
		self.patch_size = small_patterns.size(4)
		self.z_pattern = z_pattern
		self.num_patterns = small_patterns.size(1)
		self.small_patterns = small_patterns
		self.patch_only_prob = patch_only_prob 
		self.patch_loc = patch_loc
		self.sample_patch = True
		self.patch_only_scale = patch_only_scale

	def __call__(self, sample):
		start_ind = get_start_for_patch_loc(self.patch_loc, self.patch_size)
		end_ind = self.patch_size + start_ind
		img, label = sample

		# assume image is square
		pattern = np.random.randint(self.num_patterns)
		pattern_sign = -1 if np.random.random() > 0.5 else 1
		if np.random.random() > self.patch_only_prob:
			img[:,start_ind:end_ind, start_ind:end_ind] = self.z_pattern +\
				pattern_sign*self.small_patterns[label, pattern,:, :, :]
		else:
			img[:,:,:] = 0
			img[:,start_ind:end_ind, start_ind:end_ind] = self.z_pattern +\
				pattern_sign*self.patch_only_scale*np.random.random()*self.small_patterns[label, pattern,:,:,:]
		return img, label

# generate 2*num_patterns patterns, where positive offset and negative offset are both added
def pm_sym_pattern(num_classes, patch_size, num_patterns, z_scale=0.5, sep=0.01):
	pattern = torch.randn(3, patch_size, patch_size)
	
	z_patterns = z_scale*pattern

	small_patterns = torch.zeros((num_classes,num_patterns,3,patch_size,patch_size))
	small_patterns.uniform_(-sep, sep)

	return z_patterns, small_patterns

def create_m_trans(hparams):
	if hparams['transforms'] == 'none':
		return None
	
	if hparams['transforms'] == 'pm_sym_fixed':
		z_pattern, small_patterns = pm_sym_pattern(
			hparams['num_classes'],
			hparams['patch_size'],
			hparams['num_patterns'],
			hparams['z_scale'],
			hparams['patch_sep'])
		patch_aug = PatchAugmentFixed(
			z_pattern,
			small_patterns,
			hparams['patch_loc'],
			patch_only_prob=hparams['patch_only_prob'],
			patch_only_scale=hparams['patch_only_scale'])
		return [patch_aug]