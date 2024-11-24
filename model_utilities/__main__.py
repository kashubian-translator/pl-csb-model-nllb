import argparse
from logging import Logger

from transformers import NllbTokenizerFast, AutoModelForSeq2SeqLM

from evaluate.evaluator import ModelEvaluator
import shared.config_loader as config_loader
from hyperparameter_search.hyperparameter_searcher import HyperparameterSearcher
from shared.logger import set_up_logger
import train.data_loader as data_loader
from train.finetuner import ModelFinetuner
from translate.translator import Translator


def train_model(config: dict, logger: Logger) -> None:
    pretrained_model_name = config["MODEL"]["pretrained_model_name"]

    pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name)
    tokenizer = NllbTokenizerFast.from_pretrained(pretrained_model_name, additional_special_tokens=["csb_Latn"])
    dataset = data_loader.load_dataset(
        config["DATA"]["training_data_file"],
        config["DATA"]["validation_data_file"],
        config["DATA"]["test_data_file"]
    )

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
    validation_data = data_loader.load_data(config["DATA"]["validation_data_file"])

    source_data = validation_data[validation_data.columns[0]]
    target_data = validation_data[validation_data.columns[1]]

    evaluator = ModelEvaluator(logger, model, tokenizer)

    bleu, chrfpp = evaluator.evaluate(sentences=source_data, references=target_data)
    logger.info(f"Results ({source_data.name} → {target_data.name}):")
    logger.info(bleu)
    logger.info(chrfpp)

    bleu, chrfpp = evaluator.evaluate(sentences=target_data, references=source_data)
    logger.info(f"Results ({target_data.name} → {source_data.name}):")
    logger.info(bleu)
    logger.info(chrfpp)


def hyperparameter_search(config: dict, logger: Logger) -> None:
    tokenizer = NllbTokenizerFast.from_pretrained(config["MODEL"]["pretrained_model_name"], additional_special_tokens=["csb_Latn"])
    dataset = data_loader.load_dataset(
        config["DATA"]["training_data_file"],
        config["DATA"]["validation_data_file"],
        config["DATA"]["test_data_file"]
    )
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--reverse", help="Reverse translation direction", action="store_true")
    parser.add_argument("mode", choices=["train", "translate", "evaluate", "hyperparameter_search"], help="Mode to run the application with")
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
