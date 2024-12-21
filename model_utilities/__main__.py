import argparse
from logging import Logger

from transformers import NllbTokenizerFast, AutoModelForSeq2SeqLM

from evaluate.evaluator import ModelEvaluator
import shared.config_loader as config_loader
from hyperparameter_search.hyperparameter_searcher import HyperparameterSearcher
from hyperparameter_search.weight_searcher import WeightSearcher
from shared.logger import set_up_logger
import train.data_loader as data_loader
from train.finetuner import ModelFinetuner
from translate.translator import Translator


def train_model(config: dict, logger: Logger) -> None:
    pretrained_model_name = config["MODEL"]["pretrained_model_name"]

    pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name)
    tokenizer = NllbTokenizerFast.from_pretrained(pretrained_model_name, additional_special_tokens=["csb_Latn"])

    data_directory = config["DIRECTORIES"]["preprocessed_data_dir"]
    shuffle_seed = int(config["TRAINING"]["shuffle_seed"])
    weights = {item[0]: float(item[1]) for item in config.items("DATA_WEIGHTS")}

    dataset = data_loader.prepare_train_dataset(data_directory, weights, shuffle_seed)

    validation_bleu_data = data_loader.load_data(config["DATA"]["validation_bleu_data_file"])

    ModelFinetuner(logger).finetune(pretrained_model, tokenizer, dataset, config)


def translate_with_model(config: dict, logger: Logger, text: str, reverse: bool) -> None:
    output_model_name = config["MODEL"]["output_model_name"]

    model = AutoModelForSeq2SeqLM.from_pretrained(output_model_name)
    tokenizer = NllbTokenizerFast.from_pretrained(output_model_name)

    source_lang = "pol_Latn"
    target_lang = "csb_Latn"

    if reverse:
        source_lang, target_lang = target_lang, source_lang

    translation_text = Translator(logger, model, tokenizer).translate([text], source_lang, target_lang)[0]
    logger.info(f"Translation: '{text}' ({source_lang}) → '{translation_text}' ({target_lang})")


def evaluate_model(config: dict, logger: Logger) -> None:
    output_model_name = config["MODEL"]["output_model_name"]

    model = AutoModelForSeq2SeqLM.from_pretrained(output_model_name)
    tokenizer = NllbTokenizerFast.from_pretrained(output_model_name)

    evaluator = ModelEvaluator(logger, model, tokenizer)

    bleu_pol_to_csb, chrfpp_pol_to_csb, bleu_csb_to_pol, chrfpp_csb_to_pol = (
        evaluator.evaluate_dataset(config["DATA"]["validation_bleu_data_file"]))

    logger.info(f"Results (pol → csb):")
    logger.info(bleu_pol_to_csb)
    logger.info(chrfpp_pol_to_csb)

    logger.info(f"Results (csb → pol):")
    logger.info(bleu_csb_to_pol)
    logger.info(chrfpp_csb_to_pol)


def debug(config: dict, logger: Logger):
    output_model_name = config["MODEL"]["output_model_name"]

    data_loader.prepare_train_dataset(config)


def hyperparameter_search(config: dict, logger: Logger) -> None:
    tokenizer = NllbTokenizerFast.from_pretrained(config["MODEL"]["pretrained_model_name"], additional_special_tokens=["csb_Latn"])
    data_directory = config["DIRECTORIES"]["preprocessed_data_dir"]
    shuffle_seed = int(config["TRAINING"]["shuffle_seed"])
    weights = {item[0]: float(item[1]) for item in config.items("DATA_WEIGHTS")}
    dataset = data_loader.prepare_train_dataset(data_directory, weights, shuffle_seed)
    hyperparameter_space = {
        "optimizer": ["Adafactor"],
        "lr": [1e-3, 1e-4],
        "relative_step": [False],
        "clip_threshold": [0.8, 0.9],
        "decay_rate": [-0.8, -0.7],
        "weight_decay": [1e-3, 1e-2],
    }
    HyperparameterSearcher(logger, ModelFinetuner(logger)).hyperparameter_search(tokenizer, dataset, config,
                                                                                 hyperparameter_space)


def weight_search(config: dict, logger: Logger) -> None:
    pretrained_model_name = config["MODEL"]["pretrained_model_name"]
    pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name)
    tokenizer = NllbTokenizerFast.from_pretrained(pretrained_model_name, additional_special_tokens=["csb_Latn"])
    WeightSearcher(logger, ModelFinetuner(logger), config, pretrained_model, tokenizer).weight_search()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--reverse", help="Reverse translation direction", action="store_true")
    parser.add_argument("mode", choices=["train", "translate", "evaluate", "hyperparameter_search", "debug", "weight_search"], help="Mode to run the application with")
    parser.add_argument("text", type=str, nargs="?", default="Wsiądźmy do tego autobusu", help="Text to translate")

    args = parser.parse_args()

    logger = set_up_logger(__name__)

    config = config_loader.load()

    match args.mode:
        case "train":
            train_model(config, logger)
        case "translate":
            translate_with_model(config, logger, args.text, args.reverse)
        case "evaluate":
            evaluate_model(config, logger)
        case "hyperparameter_search":
            hyperparameter_search(config, logger)
        case "debug":
            debug(config, logger)
        case "weight_search":
            weight_search(config, logger)
