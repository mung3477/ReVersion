import logging
import os
from pathlib import Path
from typing import List

import numpy as np
from accelerate.logging import get_logger
from diffusers.utils import logging as diff_logging
from huggingface_hub import Repository, create_repo
from PIL.Image import Image
from transformers.utils import logging as trnsfrmrs_logging

from ..lib.path import get_full_repo_name


def config_logging(accelerator):
	logger = get_logger(__name__)
	# Make one log on every process with the configuration for debugging.
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO,
	)
	logger.info(accelerator.state, main_process_only=False)

	if accelerator.is_local_main_process:
		trnsfrmrs_logging.set_verbosity_warning()
		diff_logging.set_verbosity_info()
	else:
		trnsfrmrs_logging.set_verbosity_error()
		diff_logging.set_verbosity_error()

	return logger

def create_save_path(args, accelerator):
	def get_repo_name():
		if args.hub_model_id is not None:
			return args.hub_model_id

		return get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)

	def create_gitignore(ignore_fp: str):
		with open(ignore_fp, "w+") as gitignore:
			if "step_*" not in gitignore:
				gitignore.write("step_*\n")
			if "epoch_*" not in gitignore:
				gitignore.write("epoch_*\n")

	if accelerator.is_main_process:
		if args.push_to_hub:
			repo_name = get_repo_name()
			create_repo(repo_name, exist_ok=True, token=args.hub_token
)
			repo = Repository(
				args.output_dir,
				clone_from=repo_name,
				token=args.hub_token
			)
			create_gitignore(os.path.join(args.output_dir, ".gitignore"))
			return repo

		elif args.output_dir is not None:
			os.makedirs(args.output_dir, exist_ok=True)
			return None

def log_val_result(tracker, epoch: int, val_prompt: str,  images: List[Image]):
	if tracker.name == "tensorboard":
		np_images = np.stack([np.asarray(img) for img in images])
		tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")

	if tracker.name == "wandb":
		import wandb

		tracker.log({
			"validation": [
				wandb.Image(image, caption=f"{i}: {val_prompt}")
				for i, image in enumerate(images)
			]
		})
