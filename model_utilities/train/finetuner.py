import gc
from configparser import ConfigParser
from logging import Logger
from typing import Tuple, List, Dict, Any

import datasets as ds
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from distutils.util import strtobool
from shared.optimizer_utils import get_optimizer_class
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import trange
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM, get_constant_schedule_with_warmup, NllbTokenizerFast, \
    DataCollatorForSeq2Seq, BatchEncoding

from evaluate.evaluator import ModelEvaluator


class ModelFinetuner:
    __logger: Logger
    __evaluator: ModelEvaluator

    src_lang: str
    tgt_lang: str

    def __init__(self, logger, evaluator) -> None:
        self.__logger = logger
        self.__evaluator = evaluator
        self.src_lang = "pol_Latn"
        self.tgt_lang = "csb_Latn"

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

    def __plot_losses(self, training_losses: list[float], validation_losses: list[float]) -> None:
        plt.plot(training_losses, label="Training Loss")
        plt.plot(validation_losses, label="Validation Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training and validation losses")
        plt.legend()
        plt.savefig("./debug/graphs/losses.png")

    def __plot_translation_scores(self, translation_scores):
        plt.plot(translation_scores["bleu_pol_to_csb"],
                 label="BLEU pol→csb",
                 color="blue",
                 linestyle="-")
        plt.plot(translation_scores["bleu_csb_to_pol"],
                 label="BLEU csb→pol",
                 color="blue",
                 linestyle="--")

        plt.plot(translation_scores["chrfpp_pol_to_csb"],
                 label="chrF++ pol→csb",
                 color="red",
                 linestyle="-")
        plt.plot(translation_scores["chrfpp_csb_to_pol"],
                 label="chrF++ csb→pol",
                 color="red",
                 linestyle="--")

        plt.ylim(0, 100)
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("BLEU and chrF++ scores")
        plt.legend()

        plt.savefig("./debug/graphs/bleu_and_chrfpp.png")

    def __tokenize_dataset(self, dataset: ds.Dataset, tokenizer: NllbTokenizerFast, max_length: int) -> Dataset:
        # This function expects a dataset with 2 columns named after source and target language
        # (pol_Latn and csb_Latn in our case)
        # It should output a new dataset containing 3 columns:
        # input_ids: tokenized input sentence
        # attention_mask: attention_mask for input sentence
        # labels: tokenized target sentence (what should the model generate given the input)
        # These 3 columns correspond 1:1 to input parameters of the model.__call__() function
        # allowing us to directly unwrap each row by calling model(**row) or model(**batch) for batched input
        # Because we are training the model to work both ways, the returned dataset contains twice as many rows
        # One row for first language as input and second language as labels and another row for the other direction

        def tokenize_fn(dataset: ds.Dataset, src_lang: str, tgt_lang: str, tokenizer: NllbTokenizerFast) -> BatchEncoding:
            tokenizer.src_lang = src_lang
            tokenizer.tgt_lang = tgt_lang
            tokenized_input = tokenizer(dataset[src_lang], text_target=dataset[tgt_lang], return_tensors='pt', padding=True, truncation=True, max_length=max_length)

            # -100 is the default value ignored by pytorch
            # TODO: Check performance after removing this line
            tokenized_input.labels[tokenized_input.labels == tokenizer.pad_token_id] = -100
            return tokenized_input

        # Tokenize each row, returned dict is added to existing dataset as new columns since the 3 new keys aren't in it
        tokenized_train_dataset_pol_to_csb = dataset.map(
            lambda row: tokenize_fn(row, self.src_lang, self.tgt_lang, tokenizer),
            batched=True
        )
        tokenized_train_dataset_csb_to_pol = dataset.map(
            lambda row: tokenize_fn(row, self.tgt_lang, self.src_lang, tokenizer),
            batched=True
        )
        # Merge the two datasets together
        # TODO: Test if concatenation has better performance
        tokenized_dataset = ds.interleave_datasets([tokenized_train_dataset_pol_to_csb, tokenized_train_dataset_csb_to_pol])
        # Remove the old columns as they are not needed anymore
        tokenized_dataset = tokenized_dataset.remove_columns([self.src_lang, self.tgt_lang])

        return tokenized_dataset

    def __train_and_return_losses(self, model: AutoModelForSeq2SeqLM, tokenizer: NllbTokenizer, dataset: ds.Dataset,
                                  optimizer: Optimizer, scheduler: LambdaLR, config: ConfigParser) -> tuple[
            list[float], list[float], dict[str, list[Any]]]:
        self.__log_train_config(config)

        # Prepare variables
        train_conf = config["TRAINING"]
        batch_size = int(train_conf["batch_size"])
        max_length = int(train_conf["max_length"])
        num_epochs = int(train_conf["num_epochs"])
        shuffle_seed = int(train_conf["shuffle_seed"])
        patience = int(train_conf.get("early_stop_patience_in_epochs"))
        best_val_loss = float("inf")
        patience_counter = 0

        training_losses = []
        validation_losses = []
        translation_scores = {
            "bleu_pol_to_csb": [],
            "chrfpp_pol_to_csb": [],
            "bleu_csb_to_pol": [],
            "chrfpp_csb_to_pol": []
        }

        # Prepare dataset
        dataset = dataset.shuffle(seed=shuffle_seed)

        tokenized_train_dataset = self.__tokenize_dataset(dataset["train"], tokenizer, max_length=max_length)
        tokenized_validation_dataset = self.__tokenize_dataset(dataset["validation"], tokenizer, max_length=max_length)

        data_collator = DataCollatorForSeq2Seq(tokenizer, max_length=batch_size)

        train_dataloader = DataLoader(tokenized_train_dataset, batch_size=batch_size, collate_fn=data_collator)
        validation_dataloader = DataLoader(tokenized_validation_dataset, batch_size=batch_size, collate_fn=data_collator)

        self.__logger.debug("Starting the training process")
        model.train()
        loss = None
        self.__cleanup()

        train_batch_num = len(train_dataloader)
        val_batch_num = len(validation_dataloader)

        train_steps = num_epochs * train_batch_num

        self.__logger.debug(f"Number of epochs: {num_epochs}")
        self.__logger.debug(f"Batch number per epoch: {train_batch_num}")
        self.__logger.debug(f"Batch size: {batch_size}")
        self.__logger.debug(f"TOTAL TRAINING STEPS: {train_steps}")

        progress_bar = trange(train_steps)
        for epoch in range(num_epochs):
            epoch_training_losses = []
            for batch in train_dataloader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                try:
                    loss = model(**batch).loss
                    loss.backward()
                    epoch_training_losses.append(loss.item())

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                except Exception as e:
                    optimizer.zero_grad(set_to_none=True)
                    loss = None, None, None
                    self.__cleanup()
                    self.__logger.error("Error: unexpected exception during training, exception: %s", str(e))
                    continue

                progress_bar.update()

            avg_training_loss = sum(epoch_training_losses) / len(epoch_training_losses)
            training_losses.append(avg_training_loss)

            epoch_validation_losses = []
            with torch.no_grad():
                self.__logger.info(f"Epoch {epoch + 1}/{num_epochs}: Starting validation")
                val_progress_bar = trange(val_batch_num)
                for batch in validation_dataloader:
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    try:
                        val_loss = model(**batch).loss
                        epoch_validation_losses.append(val_loss.item())
                    except Exception as e:
                        self.__logger.error("Error: unexpected exception during validation, exception: %s", str(e))
                        continue

                    val_progress_bar.update()

            avg_validation_loss = sum(epoch_validation_losses) / len(epoch_validation_losses)
            validation_losses.append(avg_validation_loss)

            bleu_pol_to_csb, chrfpp_pol_to_csb, bleu_csb_to_pol, chrfpp_csb_to_pol = (
                self.__evaluator.evaluate_dataset(config["DATA"]["validation_bleu_data_file"]))

            translation_scores["bleu_pol_to_csb"].append(bleu_pol_to_csb.score)
            translation_scores["chrfpp_pol_to_csb"].append(chrfpp_pol_to_csb.score)
            translation_scores["bleu_csb_to_pol"].append(bleu_csb_to_pol.score)
            translation_scores["chrfpp_csb_to_pol"].append(chrfpp_csb_to_pol.score)

            self.__logger.info(f"Epoch {epoch + 1}/{num_epochs}: Training Loss: {avg_training_loss:.4f}, Validation "
                               f"Loss: {avg_validation_loss:.4f}")

            if avg_validation_loss < best_val_loss:
                best_val_loss = avg_validation_loss
                patience_counter = 0
                output_model_name = config["MODEL"]["output_model_name"]
                model.save_pretrained(output_model_name)
                tokenizer.save_pretrained(output_model_name)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.__logger.info("Early stopping triggered")
                    break

        self.__plot_losses(training_losses, validation_losses)
        self.__plot_translation_scores(translation_scores)
        return training_losses, validation_losses, translation_scores

    def finetune(self, model: AutoModelForSeq2SeqLM, tokenizer: NllbTokenizer, dataset: ds.Dataset,
                 config: ConfigParser, optimizer=None) -> tuple[list[float], list[float], dict[str, list[Any]]]:
        if torch.cuda.is_available():
            self.__logger.info("CUDA is available. Using GPU for training")
            model.cuda()
        else:
            self.__logger.info("CUDA is not available. Using CPU for training")

        if not optimizer:
            optimizer = self.__initialize_optimizer(model, config)
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(config["TRAINING"]["warmup_steps"]))

        return self.__train_and_return_losses(model, tokenizer, dataset, optimizer, scheduler, config)

    def __initialize_optimizer(self, model: AutoModelForSeq2SeqLM, config: ConfigParser):
        try:
            optimizer_name = config["OPTIMIZER"]["class_name"]
            optimizer_class = get_optimizer_class(optimizer_name)
            learning_rate = float(config["OPTIMIZER"]["learning_rate"])
            relative_step = strtobool(config["OPTIMIZER"]["relative_step"])
            clip_threshold = float(config["OPTIMIZER"]["clip_threshold"])
            decay_rate = float(config["OPTIMIZER"]["decay_rate"])
            weight_decay = float(config["OPTIMIZER"]["weight_decay"])
            scale_parameter = strtobool(config["OPTIMIZER"]["scale_parameter"])

            self.__logger.debug("Initializing optimizer with the following parameters:")
            self.__logger.debug(f"Optimizer class: {optimizer_name}")
            self.__logger.debug(f"Learning rate: {learning_rate}")
            self.__logger.debug(f"Clip threshold: {clip_threshold}")
            self.__logger.debug(f"Decay rate: {decay_rate}")
            self.__logger.debug(f"Weight decay: {weight_decay}")
            self.__logger.debug(f"Scale parameter: {scale_parameter}")
            self.__logger.debug(f"Relative step: {relative_step}")

            optimizer_params = {
                "lr": learning_rate,
                "clip_threshold": clip_threshold,
                "weight_decay": weight_decay,
                "scale_parameter": scale_parameter,
                "relative_step": relative_step
            }

            optimizer = optimizer_class([p for p in model.parameters() if p.requires_grad], **optimizer_params)
        except Exception as e:
            self.__logger.error(f"Error occurred while initializing optimizer: {e}")
            raise
        return optimizer
