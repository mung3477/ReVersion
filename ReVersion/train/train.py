import math
import os
from argparse import Namespace
from ast import Tuple
from typing import Any, TypedDict, Union

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter
from accelerate.utils import set_seed
from diffusers import (AutoencoderKL, DDPMScheduler, DiffusionPipeline,
                       DPMSolverMultistepScheduler, StableDiffusionPipeline,
                       UNet2DConditionModel)
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import (is_wandb_available,
                                          is_xformers_available)
from huggingface_hub import Repository
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, PreTrainedModel

from ..templates import relation_words, stop_words
from .args import get_chkpt_path, parse_args
from .dataset import ReVersionDataset
from .importance_sampling import Imp_smpling
from .logger import config_logging, create_save_path, log_val_result
from .steer_loss import calculate_steer_loss


class TrainSchedule(TypedDict):
	num_update_steps_per_epoch: int
	max_train_steps: int
	num_train_epochs: int

class TrainSteps(TypedDict):
	global_step: int
	first_epoch: int
	resume_step: Union[int, None]

class ReVersionTrainer():
	args: Namespace
	accelerator: Accelerator
	repo: Union[Repository, None]
	logger: MultiProcessAdapter

	stop_ids: list
	special_ids: list
	placeholder_token_id: int

	tokenizer: CLIPTokenizer
	text_encoder: PreTrainedModel
	vae: Any
	noise_scheduler: DDPMScheduler
	unet: Any

	optimizer: Optimizer
	train_dataset: ReVersionDataset
	train_dataloader: DataLoader
	train_schedule: TrainSchedule
	lr_scheduler: LambdaLR
	weight_dtype: torch.dtype
	imp_smpling: Union[Imp_smpling, None]

	progress_bar: tqdm

	def __init__(self):
		args = parse_args()
		logging_dir = os.path.join(args.output_dir, args.logging_dir)
		accelerator = Accelerator(
			gradient_accumulation_steps=args.gradient_accumulation_steps,
			mixed_precision=args.mixed_precision,
			log_with=args.report_to,
			project_dir=logging_dir,  # logging_dir=logging_dir, # depends on accelerator vesion
		)

		self.args = args
		self.accelerator = accelerator

		self.setup()

	"""
		train setup methods
	"""
	def setup(self):
		self._config()
		self._load_modules()
		self._add_tokens()
		self._config_modules()
		self._prepare_train_components()
		self._cast_types()
		self._use_importance_sampling()

		# We need to initialize the trackers we use, and also store our configuration.
		# The trackers initializes automatically on the main process.
		if self.accelerator.is_main_process:
			self.accelerator.init_trackers("textual_inversion", config=vars(self.args))

	def _config(self):
		if self.args.report_to == "wandb" and not is_wandb_available():
			raise ImportError(
				"Make sure to install wandb if you want to use it for logging during training."
			)

		# If passed along, set the training seed now.
		if self.args.seed is not None:
			set_seed(self.args.seed)

		self.logger = config_logging(self.accelerator)
		self.repo = create_save_path(self.args, self.accelerator)

	def _load_modules(self):
		# Load tokenizer
		if self.args.tokenizer_name:
			self.tokenizer = CLIPTokenizer.from_pretrained(self.args.tokenizer_name)
		elif self.args.pretrained_model_name_or_path:
			self.tokenizer = CLIPTokenizer.from_pretrained(
				self.args.pretrained_model_name_or_path, subfolder="tokenizer")

		# Load scheduler and models
		self.text_encoder = CLIPTextModel.from_pretrained(
			self.args.pretrained_model_name_or_path,
			subfolder="text_encoder",
			revision=self.args.revision
		)
		self.noise_scheduler = DDPMScheduler.from_pretrained(
			self.args.pretrained_model_name_or_path,
			subfolder="scheduler"
		)
		self.vae = AutoencoderKL.from_pretrained(
			self.args.pretrained_model_name_or_path,
			subfolder="vae",
			revision=self.args.revision
		)
		self.unet = UNet2DConditionModel.from_pretrained(
			self.args.pretrained_model_name_or_path,
			subfolder="unet",
			revision=self.args.revision
		)

	def _add_tokens(self):
		tokenizer = self.tokenizer
		placeholder_token = self.args.placeholder_token

		# Add the placeholder token in tokenizer
		num_added_tokens = tokenizer.add_tokens(placeholder_token)
		assert num_added_tokens > 0, f"The tokenizer already contains the token {placeholder_token}. Please pass a different `placeholder_token` that is not already in the tokenizer."

		# Convert the initializer_token, placeholder_token to ids
		# Check if initializer_token is a single token or a sequence of tokens
		token_ids = tokenizer.encode(
			self.args.initializer_token,
			add_special_tokens=False
		)
		assert len(token_ids) == 1, "The initializer token must be a single token."

		initializer_token_id = token_ids[0]
		placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

		# stop words id
		expanded_stop_words = stop_words + relation_words  # add relation words to stop_words
		stop_ids = tokenizer(
			" ".join(expanded_stop_words),
			truncation=False,
			return_tensors="pt",
		).input_ids[0].detach().tolist()

		# # add special token ids to stop ids
		# stop_ids = stop_ids + [tokenizer.bos_token_id, tokenizer.eos_token_id]
		special_ids = [tokenizer.bos_token_id, tokenizer.eos_token_id]

		# Resize the token embeddings as we are adding new special tokens to the tokenizer
		self.text_encoder.resize_token_embeddings(len(tokenizer))

		# Initialise the newly added placeholder token with the embeddings of the initializer token
		token_embeds = self.text_encoder.get_input_embeddings().weight.data
		token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

		self.stop_ids = stop_ids
		self.special_ids = special_ids
		self.placeholder_token_id = placeholder_token_id

	def _config_modules(self):
		# Freeze vae and unet
		self.vae.requires_grad_(False)
		self.unet.requires_grad_(False)

		# Freeze all parameters except for the token embeddings in text encoder
		self.text_encoder.text_model.encoder.requires_grad_(False)
		self.text_encoder.text_model.final_layer_norm.requires_grad_(False)
		self.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

		if self.args.gradient_checkpointing:
			# Keep unet in train mode if we are using gradient checkpointing to save memory.
			# The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
			self.unet.train()
			self.text_encoder.gradient_checkpointing_enable()
			self.unet.enable_gradient_checkpointing()

		if self.args.enable_xformers_memory_efficient_attention:
			assert is_xformers_available() == True, "xformers is not available. Make sure it is installed correctly"
			self.unet.enable_xformers_memory_efficient_attention()

		# Enable TF32 for faster training on Ampere GPUs,
		# cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
		if self.args.allow_tf32:
			torch.backends.cuda.matmul.allow_tf32 = True

	def _calc_train_steps_and_epochs(self, data_len: int)->TrainSchedule:
		args = self.args
		# Scheduler and math around the number of training steps.
		num_update_steps_per_epoch = math.ceil(data_len / args.gradient_accumulation_steps)

		if args.max_train_steps:
			return {
				"num_update_steps_per_epoch": num_update_steps_per_epoch,
				"max_train_steps": args.max_train_steps,
				"num_train_epochs": args.num_train_epochs
			}

		return {
			"num_update_steps_per_epoch": num_update_steps_per_epoch,
			"max_train_steps": args.num_train_epochs * num_update_steps_per_epoch,
			"num_train_epochs": math.ceil(args.max_train_steps / num_update_steps_per_epoch)
		}

	def _prepare_train_components(self):
		args = self.args
		tokenizer = self.tokenizer

		if args.scale_lr:
			args.learning_rate = (
				args.learning_rate * args.gradient_accumulation_steps *
				args.train_batch_size * self.accelerator.num_processes
			)

		# Initialize the optimizer
		optimizer = torch.optim.AdamW(
			# only optimize the embeddings
			self.text_encoder.get_input_embeddings().parameters(),
			lr=args.learning_rate,
			betas=(args.adam_beta1, args.adam_beta2),
			weight_decay=args.adam_weight_decay,
			eps=args.adam_epsilon,
		)

		# Dataset and DataLoaders creation:
		train_dataset = ReVersionDataset(
			data_root=args.train_data_dir,
			tokenizer=tokenizer,
			size=args.resolution,
			placeholder_token=args.placeholder_token,
			repeats=args.repeats,
			center_crop=args.center_crop,
			set="train",
			relation_words=relation_words,
			num_positives=args.num_positives
		)

		train_dataloader = DataLoader(
			train_dataset,
			batch_size=args.train_batch_size,
			shuffle=True,
			num_workers=args.dataloader_num_workers
		)

		train_schedule = self._calc_train_steps_and_epochs(len(train_dataloader))

		lr_scheduler = get_scheduler(
			args.lr_scheduler,
			optimizer=optimizer,
			num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
			num_training_steps=train_schedule["max_train_steps"] * args.gradient_accumulation_steps
		)

		# Prepare everything with our `accelerator`.
		text_encoder, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
			self.text_encoder, optimizer, train_dataloader, lr_scheduler
		)

		self.text_encoder = text_encoder
		self.optimizer = optimizer
		self.train_dataset = train_dataset
		self.train_dataloader = train_dataloader
		self.lr_scheduler = lr_scheduler
		self.train_schedule = train_schedule

	def _cast_types(self):
		"""
			For mixed precision training,
			we cast the unet and vae weights to half-precision as these models are only used for inference.
			Keeping weights in full precision is not required.
		"""
		accelerator = self.accelerator
		weight_dtype = torch.float32

		if accelerator.mixed_precision == "fp16":
			weight_dtype = torch.float16
		elif accelerator.mixed_precision == "bf16":
			weight_dtype = torch.bfloat16

		# Move vae and unet to device and cast to weight_dtype
		self.unet.to(accelerator.device, dtype=weight_dtype)
		self.vae.to(accelerator.device, dtype=weight_dtype)

		# We need to recalculate our total training steps as the size of the training dataloader may have changed.
		self.train_schedule = self._calc_train_steps_and_epochs(len(self.train_dataloader))
		self.weight_dtype = weight_dtype

	"""
		train methods
	"""
	def _use_importance_sampling(self):
		args = self.args
		noise_scheduler = self.noise_scheduler

		# Relation-Focal Importance Sampling
		if args.importance_sampling:
			print("Using Relation-Focal Importance Sampling")
			self.imp_smpling = Imp_smpling(
				num_train_timesteps=noise_scheduler.config.num_train_timesteps,
				scaled_cosine_alpha=args.scaled_cosine_alpha
			)
		else:
			self.imp_smpling = None

	def _resume_from_chckpt(self)->TrainSteps:
		args = self.args
		accelerator = self.accelerator
		num_update_steps_per_epoch = self.train_schedule["num_update_steps_per_epoch"]

		train_steps: TrainSteps = {
			"global_step": 0,
			"first_epoch": 0,
			"resume_step": None
		}

		if args.resume_from_checkpoint is None:
				return train_steps

		path = get_chkpt_path(
			resume_from_checkpoint=args.resume_from_checkpoint,
			output_dir=args.output_dir
		)

		if path is None:
			accelerator.print(
				f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
			)
			self.args.resume_from_checkpoint = None
		else:
			accelerator.print(f"Resuming from checkpoint {path}")
			accelerator.load_state(os.path.join(args.output_dir, path))

			global_step = int(path.split("-")[1])
			resume_global_step = global_step * args.gradient_accumulation_steps
			first_epoch = global_step // num_update_steps_per_epoch
			resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

			train_steps["global_step"] = global_step
			train_steps["first_epoch"] = first_epoch
			train_steps["resume_step"] = resume_step

		return train_steps

	def _print_train_start(self, global_step: int):
		# Train!
		args = self.args
		total_batch_size = args.train_batch_size * self.accelerator.num_processes * args.gradient_accumulation_steps

		self.logger.info("***** Running training *****")
		self.logger.info(f"  Num examples = {len(self.train_dataset)}")
		self.logger.info(f"  Num Epochs = {args.num_train_epochs}")
		self.logger.info(
			f"  Instantaneous batch size per device = {args.train_batch_size}")
		self.logger.info(
			f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
		)
		self.logger.info(
			f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
		self.logger.info(f"  Total optimization steps = {args.max_train_steps}")

		# Only show the progress bar once on each machine.
		progress_bar = tqdm(
			range(global_step, args.max_train_steps),
			disable=not self.accelerator.is_local_main_process
		)
		progress_bar.set_description("Steps")

		self.progress_bar = progress_bar

	def _get_model_pred_and_trgt(self, batch):
		use_imp_smpling = self.args.importance_sampling and self.imp_smpling is not None
		noise_scheduler = self.noise_scheduler
		num_train_timesteps = noise_scheduler.config.num_train_timesteps
		prediction_type = noise_scheduler.config.prediction_type
		weight_dtype = self.weight_dtype

		# Convert images to latent space
		pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
		latents = self.vae.encode(pixel_values).latent_dist.sample().detach()
		latents = latents * self.vae.config.scaling_factor

		# Sample noise that we'll add to the latents
		noise = torch.randn_like(latents)
		bsz = latents.shape[0]

		# timestep (t) sampling
		timesteps = self.imp_smpling.sample(bsz) if use_imp_smpling \
			else torch.randint(0, num_train_timesteps, (bsz, ), device=latents.device)
		timesteps = timesteps.long()

		# Add noise to the latents according to the noise magnitude at each timestep
		# (this is the forward diffusion process)
		noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

		# Get the text embedding for conditioning
		encoder_hidden_states = self.text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)

		# Predict the noise residual
		model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

		# Get the target for loss depending on the prediction type
		if prediction_type == "epsilon":
			target = noise
		elif prediction_type == "v_prediction":
			target = noise_scheduler.get_velocity(latents, noise, timesteps)
		else:
			raise ValueError(f"Unknown prediction type {prediction_type}")

		return model_pred, target

	def _calc_loss(self, pred, trgt, batch):
		"""
			return: (loss, logs)
		"""
		args = self.args
		accelerator = self.accelerator

		loss = 0.0
		use_steer_loss = args.steer_loss_weight > 0 and args.num_positives > 0

		# L_denoise
		denoise_loss = F.mse_loss(pred.float(), trgt.float(), reduction="mean")
		weighted_denoise_loss = args.denoise_loss_weight * denoise_loss
		loss += weighted_denoise_loss

		# L_steer
		if use_steer_loss:
			token_embedding = accelerator.unwrap_model(self.text_encoder).get_input_embeddings()  # with grad
			steer_loss = calculate_steer_loss(
				token_embedding,
				batch["input_ids"],
				self.placeholder_token_id,
				self.stop_ids,
				self.special_ids,
				batch["positive_ids"],
				temperature=args.temperature
			)
			weighted_steer_loss = args.steer_loss_weight * steer_loss
			loss += weighted_steer_loss

		logs = {
			"lr": self.lr_scheduler.get_last_lr()[0],
			"loss": loss.detach().item(),
			"denoise_loss": denoise_loss.detach().item(),
			"weighted_denoise_loss": weighted_denoise_loss.detach().item(),
		}
		if use_steer_loss:
			logs["steer_loss"] = steer_loss.detach().item()
			logs["weighted_steer_loss"] = weighted_steer_loss.detach().item()

		return loss, logs

	def _is_before_resume_step(self, train_steps: TrainSteps, epoch: int, step: int):
		resume_from_checkpoint = self.args.resume_from_checkpoint
		first_epoch = train_steps["first_epoch"]
		resume_step = train_steps["resume_step"]

		return resume_from_checkpoint and resume_step is not None \
				and epoch == first_epoch and step < resume_step

	def _train(self, train_steps: TrainSteps, epoch: int, orig_embeds_params):
		args = self.args
		accelerator = self.accelerator

		self.text_encoder.train()
		for step, batch in enumerate(self.train_dataloader):
			# Skip steps until we reach the resumed step
			if self._is_before_resume_step(train_steps=train_steps, epoch=epoch, step=step):
				if step % args.gradient_accumulation_steps == 0:
					self.progress_bar.update(1)
				continue

			with accelerator.accumulate(self.text_encoder):
				model_pred, target = self._get_model_pred_and_trgt(batch=batch)
				loss, loss_logs = self._calc_loss(model_pred, target, batch=batch)

				accelerator.backward(loss)
				self.optimizer.step()
				self.lr_scheduler.step()

				self.optimizer.zero_grad()

				# Let's make sure we don't update any embedding weights besides the newly added token
				index_no_updates = torch.arange(len(self.tokenizer)) != self.placeholder_token_id
				with torch.no_grad():
					accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[index_no_updates] \
						= orig_embeds_params[index_no_updates]

			self._save_progress(train_steps=train_steps)
			self.progress_bar.set_postfix(**loss_logs)
			accelerator.log(loss_logs, step=train_steps["global_step"])

			if train_steps["global_step"] >= args.max_train_steps:
				break

	def _validation(self, epoch: int):
		args = self.args
		accelerator = self.accelerator
		val_prompt = args.validation_prompt
		val_num = args.num_validation_images

		def _load_pipe():
			# create pipeline (note: unet and vae are loaded again in float32)
			pipeline = DiffusionPipeline.from_pretrained(
				args.pretrained_model_name_or_path,
				text_encoder=accelerator.unwrap_model(self.text_encoder),
				tokenizer=self.tokenizer,
				unet=self.unet,
				vae=self.vae,
				revision=args.revision,
				torch_dtype=self.weight_dtype,
			)
			pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
				pipeline.scheduler.config)
			pipeline = pipeline.to(accelerator.device)
			pipeline.set_progress_bar_config(disable=True)

			return pipeline

		def _run_inference(pipeline):
			generator = None if args.seed is None \
								else torch.Generator(device=accelerator.device).manual_seed(args.seed)
			images = []
			for _ in range(val_num):
				with torch.autocast("cuda"):
					image = pipeline(
						val_prompt,
						num_inference_steps=25,
						generator=generator).images[0]
				images.append(image)

			return images

		if val_prompt is not None and epoch % args.validation_epochs == 0:
			self.logger.info(
				f"Running validation... \n Generating {val_num} images with prompt: {val_prompt}."
			)

			# run inference
			pipeline = _load_pipe()
			images = _run_inference(pipeline=pipeline)

			for tracker in accelerator.trackers:
				log_val_result(tracker, epoch=epoch, val_prompt=val_prompt, images=images)
			del pipeline
			torch.cuda.empty_cache()

	"""
		logging methods
	"""
	def _save_embeddings(self, save_path: str):
		learned_embeds = self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[self.placeholder_token_id]
		learned_embeds_dict = {
			self.args.placeholder_token: learned_embeds.detach().cpu()
		}
		torch.save(learned_embeds_dict, save_path)

	def _save_progress(self, train_steps: TrainSteps):
		args = self.args
		accelerator = self.accelerator

		def _save_step():
			self.logger.info("Saving embeddings")
			save_path = os.path.join(args.output_dir, f"learned_embeds-steps-{global_step}.bin")
			self._save_embeddings(save_path=save_path)

		def _save_chkpt():
			if accelerator.is_main_process:
					save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
					accelerator.save_state(save_path)
					self.logger.info(f"Saved state to {save_path}")

		# Checks if the accelerator has performed an optimization step behind the scenes
		if accelerator.sync_gradients:
			self.progress_bar.update(1)
			global_step = train_steps["global_step"] + 1
			if global_step % args.save_steps == 0:
				_save_step()

			if global_step % args.checkpointing_steps == 0:
				_save_chkpt()

			train_steps["global_step"] = global_step

	def _save_train_result(self):
		args = self.args
		accelerator = self.accelerator

		def _does_save_full_model():
			if args.push_to_hub and args.only_save_embeds:
				self.logger.warn("Enabling full model saving because --push_to_hub=True was specified.")
			return not args.only_save_embeds or args.push_to_hub

		# Create the pipeline using using the trained modules and save it.
		accelerator.wait_for_everyone()
		if accelerator.is_main_process:
			if _does_save_full_model():
				pipeline = StableDiffusionPipeline.from_pretrained(
					args.pretrained_model_name_or_path,
					text_encoder=accelerator.unwrap_model(self.text_encoder),
					vae=self.vae,
					unet=self.unet,
					tokenizer=self.tokenizer,
				)
				pipeline.save_pretrained(args.output_dir)

			# Save the newly trained embeddings
			save_path = os.path.join(args.output_dir, "learned_embeds.bin")
			self._save_embeddings(self.text_encoder, self.placeholder_token_id, accelerator, args, save_path)

			if args.push_to_hub:
				self.repo.push_to_hub(
					commit_message="End of training",
					blocking=False,
					auto_lfs_prune=True)

		accelerator.end_training()

	def forward(self):
		args = self.args
		accelerator = self.accelerator
		text_encoder = self.text_encoder
		train_steps = self._resume_from_chckpt()
		self._print_train_start(global_step=train_steps["global_step"])

		# keep original embeddings as reference
		orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()

		for epoch in range(train_steps["first_epoch"], args.num_train_epochs):
			self._train(train_steps=train_steps, epoch=epoch, orig_embeds_params=orig_embeds_params)
			self._validation(epoch=epoch)

		self._save_train_result()
