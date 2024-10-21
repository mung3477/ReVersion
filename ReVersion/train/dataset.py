import json
import os
import random
from typing import List, Optional

import numpy as np
import PIL
import PIL.Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ReVersionDataset(Dataset):
	PIL_INTERPOLATION: dict
	IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif')

	@classmethod
	def define_interpolations(cls):
		from packaging import version

		if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
			cls.PIL_INTERPOLATION = {
				"linear": PIL.Image.Resampling.BILINEAR,
				"bilinear": PIL.Image.Resampling.BILINEAR,
				"bicubic": PIL.Image.Resampling.BICUBIC,
				"lanczos": PIL.Image.Resampling.LANCZOS,
				"nearest": PIL.Image.Resampling.NEAREST,
			}
		else:
			cls.PIL_INTERPOLATION = {
				"linear": PIL.Image.LINEAR,
				"bilinear": PIL.Image.BILINEAR,
				"bicubic": PIL.Image.BICUBIC,
				"lanczos": PIL.Image.LANCZOS,
				"nearest": PIL.Image.NEAREST,
			}


	@classmethod
	def is_image_file(cls, filename: str):
		# return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
		return filename.endswith(cls.IMG_EXTENSIONS)

	def __init__(
		self,
		data_root,
		tokenizer,
		size=512,
		repeats=100,
		interpolation="bicubic",
		flip_p=0.0,  # do not flip horizontally, otherwise might affect the relation
		set="train",
		placeholder_token="*",
		center_crop=False,
		relation_words: Optional[List[str]] = None,
		num_positives=1,
	):
		self.data_root = data_root
		self.define_interpolations()

		# read per image templates
		local_f = open(os.path.join(data_root, 'text.json'))
		self.templates = json.load(local_f)
		print(f'self.templates={self.templates}')

		self.tokenizer = tokenizer
		self.size = size
		self.placeholder_token = placeholder_token
		self.center_crop = center_crop
		self.flip_p = flip_p

		# for Relation-Steering
		self.relation_words = relation_words
		self.num_positives = num_positives

		# record image paths
		self.image_paths = []
		for file_path in os.listdir(self.data_root):
			# if file_path != 'text.json':

			if self.is_image_file(file_path):
				self.image_paths.append(
					os.path.join(self.data_root, file_path))

		self.num_images = len(self.image_paths)
		self._length = self.num_images

		if set == "train":
			self._length = self.num_images * repeats

		self.interpolation = {
			"linear": self.PIL_INTERPOLATION["linear"],
			"bilinear": self.PIL_INTERPOLATION["bilinear"],
			"bicubic": self.PIL_INTERPOLATION["bicubic"],
			"lanczos": self.PIL_INTERPOLATION["lanczos"],
		}[interpolation]

		self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

	def __len__(self):
		return self._length

	def __getitem__(self, i):


		example = {}

		# exemplar images
		image_path = self.image_paths[i % self.num_images]
		image = PIL.Image.open(image_path)
		image_name = image_path.split('/')[-1]

		if not image.mode == "RGB":
			image = image.convert("RGB")

		placeholder_string = self.placeholder_token

		# coarse descriptions
		text = random.choice(
			self.templates[image_name]).format(placeholder_string)

		example["input_ids"] = self.tokenizer(
			text,
			padding="max_length",
			truncation=True,
			max_length=self.tokenizer.model_max_length,
			return_tensors="pt",
		).input_ids[0]

		# randomly sample positive words for L_steer
		if self.num_positives > 0:
			assert self.relation_words is not None, f"Empty population for sampling: {self.relation_words}"

			positive_words = random.sample(self.relation_words, k=self.num_positives)
			positive_words_string = " ".join(positive_words)
			example["positive_ids"] = self.tokenizer(
				positive_words_string,
				padding="max_length",
				truncation=True,
				max_length=self.tokenizer.model_max_length,
				return_tensors="pt",
			).input_ids[0]

		# default to score-sde preprocessing
		img = np.array(image).astype(np.uint8)

		if self.center_crop:
			crop = min(img.shape[0], img.shape[1])
			(
				h,
				w,
			) = (
				img.shape[0],
				img.shape[1],
			)
			img = img[(h - crop) // 2:(h + crop) // 2,
					  (w - crop) // 2:(w + crop) // 2]

		image = PIL.Image.fromarray(img)
		image = image.resize((self.size, self.size),
							 resample=self.interpolation)

		image = self.flip_transform(image)
		image = np.array(image).astype(np.uint8)
		image = (image / 127.5 - 1.0).astype(np.float32)

		example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)

		return example

