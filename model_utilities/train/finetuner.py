import gc
import random
from logging import Logger

import matplotlib.pyplot as plt
import torch
import pandas as pd
import datasets as ds
from tqdm.auto import trange
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM, get_constant_schedule_with_warmup
from transformers.optimization import Adafactor

from configparser import ConfigParser


class ModelFinetuner:
    __logger: Logger

    def __init__(self, logger) -> None:
        self.__logger = logger

    def __cleanup(self) -> None:
        """Try to free GPU memory"""
        gc.collect()
        torch.cuda.empty_cache()

    def __log_train_config(self, config: ConfigParser) -> None:
        self.__logger.info("=" * 40)
        self.__logger.info("CONFIGURATION SETTINGS")
        self.__logger.info("=" * 40)
        for section in config.sections():
            self.__logger.info(f"[{section}]")
            for key, value in config.items(section):
                self.__logger.info(f"{key}: {value}")
            self.__logger.info("-" * 40)
        self.__logger.info("=" * 40 + "\n")

    def __plot_losses(self, losses: list[float]) -> None:
        plt.plot(losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.savefig("./debug/graphs/losses.png")

    def __train(self, model: AutoModelForSeq2SeqLM, tokenizer: NllbTokenizer, dataset: ds.Dataset, optimizer: Adafactor, config: ConfigParser) -> None:
        self.__log_train_config(config)

        train_conf = config["TRAINING"]

        losses = []
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(train_conf["warmup_steps"]))

        batched_dataset = dataset["train"].batch(batch_size=int(train_conf["batch_size"]))

        self.__logger.debug("Starting the training process")
        model.train()
        x, y, loss = None, None, None
        self.__cleanup()

        num_epochs = int(train_conf["num_epochs"])
        num_training_steps = len(batched_dataset)

        self.__logger.debug(f"Training steps per epoch: {num_training_steps}")

        progress_bar = trange(num_epochs * num_training_steps)
        for _ in range(num_epochs):
            for i in range(num_training_steps):
                batch = batched_dataset[i]
                # Swap the direction of translation for some batches
                batch = list(batch.items())
                random.shuffle(batch)

                (lang1, xx), (lang2, yy) = batch
                try:
                    tokenizer.src_lang = lang1
                    x = tokenizer(xx, return_tensors='pt', padding=True, truncation=True, max_length=int(train_conf["max_length"])).to(model.device)
                    tokenizer.src_lang = lang2
                    y = tokenizer(yy, return_tensors='pt', padding=True, truncation=True, max_length=int(train_conf["max_length"])).to(model.device)
                    y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100

                    loss = model(**x, labels=y.input_ids).loss
                    loss.backward()
                    losses.append(loss.item())

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                except Exception as e:
                    optimizer.zero_grad(set_to_none=True)
                    x, y, loss = None, None, None
                    self.__cleanup()
                    self.__logger.error("Error: unexpected exception during training, exception: %s", str(e))
                    continue

                progress_bar.update()

        self.__plot_losses(losses)

        output_model_name = config["MODEL"]["output_model_name"]
        try:
            model.save_pretrained(output_model_name)
            tokenizer.save_pretrained(output_model_name)
        except Exception as e:
            self.__logger.error("Error: saving model/tokenizer failed, exception: %s", str(e))
            raise

    def finetune(self, model: AutoModelForSeq2SeqLM, dataset: ds.Dataset, tokenizer: NllbTokenizer, config: ConfigParser) -> None:
        if torch.cuda.is_available():
            self.__logger.info("CUDA is available. Using GPU for training")
            model.cuda()
        else:
            self.__logger.info("CUDA is not available. Using CPU for training")

        try:
            optimizer = Adafactor(
                [p for p in model.parameters() if p.requires_grad],
                scale_parameter=False,
                relative_step=False,
                lr=1e-4,
                clip_threshold=1.0,
                weight_decay=1e-3,
            )
        except Exception as e:
            self.__logger.error(f"Error occurred while initializing Adafactor: {e}")
            raise

        self.__train(model, dataset, tokenizer, optimizer, config)
