from configparser import ConfigParser
from logging import Logger

import optuna
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast

from train.data_loader import prepare_train_dataset
from train.finetuner import ModelFinetuner


class WeightSearcher:

    def __init__(self, logger: Logger, model_finetuner: ModelFinetuner, config: ConfigParser,
                 model: AutoModelForSeq2SeqLM, tokenizer: NllbTokenizerFast) -> None:
        self.__logger = logger
        self.__model_finetuner = model_finetuner
        self.__config = config
        self.__model = model
        self.__tokenizer = tokenizer

    def objective(self, trial: optuna.Trial) -> float:
        w_dictionaries = trial.suggest_float("dictionaries_weight", 0.0, 1.0)
        w_corpus = trial.suggest_float("corpus_weight", 0.0, 1.0)
        w_remus = trial.suggest_float("remus_weight", 0.0, 1.0)

        total = w_dictionaries + w_corpus + w_remus

        if total == 0:
            w_dictionaries, w_corpus, w_remus = 1 / 3, 1 / 3, 1 / 3
        else:
            w_dictionaries /= total
            w_corpus /= total
            w_remus /= total

        data_directory = self.__config["DIRECTORIES"]["preprocessed_data_dir"]
        shuffle_seed = int(self.__config["TRAINING"]["shuffle_seed"])

        current_weights = {
            "dictionaries-mini": w_dictionaries,
            "corpus-mini": w_corpus,
            "remus-mini": w_remus
        }

        dataset = prepare_train_dataset(data_directory, current_weights, shuffle_seed)

        _, _, translation_scores = self.__model_finetuner.finetune(
            self.__model,
            self.__tokenizer,
            dataset,
            self.__config
        )

        patience = int(self.__config["TRAINING"]["early_stop_patience_in_epochs"])

        final_bleu = ((translation_scores["bleu_pol_to_csb"][-patience - 1]
                      + translation_scores["bleu_csb_to_pol"][-patience - 1])
                      / 2)
        return final_bleu

    def weight_search(self) -> None:
        study = optuna.create_study(study_name="weight_search", direction="maximize")

        n_trials = int(self.__config["WEIGHT_SEARCH"]["weight_search_trials_number"])
        self.__logger.info(f"Starting weight search with n_trials={n_trials}.")
        study.optimize(self.objective, n_trials=n_trials)

        best_trial = study.best_trial
        self.__logger.info("Weight search complete.")
        self.__logger.info(f"Best trial number: {best_trial.number}")
        self.__logger.info(f"Best trial BLEU: {best_trial.value:.4f}")
        self.__logger.info(f"Best weights (unnormalized from the search): {best_trial.params}")

        w_dict = best_trial.params["dictionaries_weight"]
        w_corp = best_trial.params["corpus_weight"]
        w_rem = best_trial.params["remus_weight"]
        total = w_dict + w_corp + w_rem
        if total == 0:
            w_dict, w_corp, w_rem = 1 / 3, 1 / 3, 1 / 3
        else:
            w_dict /= total
            w_corp /= total
            w_rem /= total

        self.__logger.info(
            f"Best weight ratio -> dictionaries: {w_dict:.3f}, "
            f"corpus: {w_corp:.3f}, remus: {w_rem:.3f}"
        )
