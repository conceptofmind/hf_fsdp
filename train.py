import torch
import argparse
import os
import math
import time
import random
import wandb
import transformers
import typing

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import InitProcessGroupKwargs, set_seed
from datetime import timedelta
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
from transformers import (
    get_cosine_schedule_with_warmup,
    set_seed,
    default_data_collator,
)
from tqdm import tqdm
from together.modeling_flash_llama import LlamaForCausalLM


def main():

    parser = argparse.ArgumentParser(description='Training script for the model')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training.')
    parser.add_argument('--gradient_accumulate_every', type=int, default=1,
                        help='Gradient accumulation steps.')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to resume training from a checkpoint.')
    parser.add_argument('--checkpointing_steps', type=int, default=1000,
                        help='Steps interval for checkpointing.')
    parser.add_argument('--output_dir', type=str, default="",
                        help='Directory to save the model and output.')
    parser.add_argument('--wandb_entity', type=str, default="",
                        help='WandB entity.')
    parser.add_argument('--wandb_project', type=str, default="",
                        help='WandB project.')
    parser.add_argument('--wandb_name', type=str, default="",
                        help='WandB name.')
    parser.add_argument('--wandb_id', type=str, default=None,
                        help='WandB ID.')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf",
                        help='Model name or path to be loaded for training.')
    parser.add_argument('--dataset_name', type=str, default=DATASET_NAME,
                        help='Name or path of the dataset.')

    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    GRADIENT_ACCUMULATE_EVERY = args.gradient_accumulate_every
    RESUME_FROM_CHECKPOINT = args.resume_from_checkpoint
    CHECKPOINTING_STEPS = args.checkpointing_steps
    OUTPUT_DIR = args.output_dir
    WANDB_ENTITY = args.wandb_entity
    WANDB_PROJECT = args.wandb_project
    WANDB_NAME = args.wandb_name
    WANDB_ID = args.wandb_id
    MODEL_NAME = args.model_name
    DATASET_NAME = args.dataset_name

    set_seed(42)

    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))

    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATE_EVERY,
        mixed_precision="bf16",
        log_with="wandb",
        kwargs_handlers=[timeout],
    )

    accelerator.init_trackers(
        project_name="pubmed-llama-2",
        init_kwargs={
            "wandb": {
                "entity": WANDB_ENTITY,
                "name": WANDB_NAME,
                "id": WANDB_ID,
                "resume": "must" if WANDB_ID else None,
            }
        },
    )

    accelerator.print(f"Total GPUS: {accelerator.num_processes}")

    # Create fresh LlamaForCausalLM model
    model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )

    model = accelerator.prepare(model)

    model.gradient_checkpointing_enable()

    accelerator.print(f"FSDP model parameters per device: {model.num_parameters():,}")
    accelerator.print(
        f"Training a {accelerator.num_processes * model.num_parameters():,} parameter model"
    )

    # Dataloaders

    train_dataset = load_dataset(
        DATASET_NAME,
        num_proc=os.cpu_count() - 1,
    )["train"]

    # Optional: Select random 10% of the dataset
    # train_dataset = train.shuffle(seed=42).select(range(0, len(train), 10))
    # Optional: Repeat the dataset 3 times
    # train_dataset = concatenate_datasets([train_dataset] * 3)

    train_loader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        shuffle=True,
        batch_size=BATCH_SIZE,
    )

    # Optimizer set up

    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Determine number of training steps

    max_train_steps = math.ceil(len(train_loader) / GRADIENT_ACCUMULATE_EVERY)
    accelerator.print(f"Max train steps: {max_train_steps}")

    # Dummy Scheduler for DeepSpeed

    scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_training_steps=max_train_steps,
        num_warmup_steps=100,
    )

    # prepare

    optim, train_loader, scheduler = accelerator.prepare(optim, train_loader, scheduler)

    # checkpoint scheduler

    accelerator.register_for_checkpointing(scheduler)

    # Recalculate

    max_train_steps = math.ceil(len(train_loader) / GRADIENT_ACCUMULATE_EVERY)
    accelerator.print(f"Max train steps recalculated: {max_train_steps}")

    # Total batch size for logging

    total_batch_size = (
        BATCH_SIZE * accelerator.num_processes * GRADIENT_ACCUMULATE_EVERY
    )
    accelerator.print(f"Total batch size: {total_batch_size}")

    # Resume training

    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0

    if RESUME_FROM_CHECKPOINT:
        if RESUME_FROM_CHECKPOINT is not None or RESUME_FROM_CHECKPOINT != "":
            accelerator.print(f"Resuming from checkpoint {RESUME_FROM_CHECKPOINT}")
            accelerator.load_state(RESUME_FROM_CHECKPOINT)
            path = os.path.basename(RESUME_FROM_CHECKPOINT)
        training_difference = os.path.splitext(path)[0]

        resume_step = int(training_difference.replace("step_", ""))

    if RESUME_FROM_CHECKPOINT and resume_step is not None:
        # We need to skip steps until we reach the resumed step
        train_loader = accelerator.skip_first_batches(train_loader, resume_step)
        completed_steps += resume_step
        progress_bar.update(resume_step)
        accelerator.print(f"Resuming training from step {resume_step}")

    # Training

    model.train()
    for batch in train_loader:
        with accelerator.accumulate(model):
            loss = model(**batch).loss
            accelerator.backward(loss)

            accelerator.log({"loss": loss.item()}, step=completed_steps)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optim.step()
            scheduler.step()
            optim.zero_grad()

        accelerator.log({"lr": optim.param_groups[0]["lr"]}, step=completed_steps)

        if accelerator.sync_gradients:
            progress_bar.update(1)
            completed_steps += 1

        if isinstance(CHECKPOINTING_STEPS, int):
            if completed_steps % CHECKPOINTING_STEPS == 0:
                output_dir = f"step_{completed_steps}"
                if OUTPUT_DIR is not None:
                    output_dir = os.path.join(OUTPUT_DIR, output_dir)
                accelerator.save_state(output_dir)
                accelerator.print(f"Saving Finished")

        if completed_steps >= max_train_steps:
            break

    # end training

    accelerator.print(f"Training Finished")
    accelerator.end_training()

    # save final model

    accelerator.print(f"Saving model to {OUTPUT_DIR}")
    if OUTPUT_DIR is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        unwrapped_model.save_pretrained(
            f"{OUTPUT_DIR}/final",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )

        accelerator.print(f"Saving Finished")


if __name__ == "__main__":
    main()