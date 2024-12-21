from logging import Logger

import pandas as pd
import sacrebleu
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

from translate.translator import Translator

from train.data_loader import load_data


class ModelEvaluator:
    __logger: Logger

    def __init__(self, logger, model: AutoModelForSeq2SeqLM, tokenizer: NllbTokenizer) -> None:
        self.__logger = logger
        self.__translator = Translator(self.__logger, model, tokenizer)

    def __evaluate(self, sentences: pd.Series, references: pd.Series) -> tuple[sacrebleu.metrics.BLEUScore, sacrebleu.metrics.CHRFScore]:
        source_lang = sentences.name
        target_lang = references.name

        translated_sentences_list = self.__translator.translate(sentences.to_list(), source_lang, target_lang)

        # There can be multiple references per sentence so we need to pass a one element list
        references_list = [references.to_list()]

        bleu = sacrebleu.corpus_bleu(translated_sentences_list, references_list)
        chrfpp = sacrebleu.corpus_chrf(translated_sentences_list, references_list, word_order=2)

        return (bleu, chrfpp)

    def evaluate_dataset(self, data_filename: str):
        validation_bleu_data = load_data(data_filename)
        polish_data = validation_bleu_data[validation_bleu_data.columns[0]]
        kashubian_data = validation_bleu_data[validation_bleu_data.columns[1]]

        bleu_pol_to_csb, chrfpp_pol_to_csb = self.__evaluate(sentences=polish_data, references=kashubian_data)
        bleu_csb_to_pol, chrfpp_csb_to_pol = self.__evaluate(sentences=kashubian_data, references=polish_data)

        return bleu_pol_to_csb, chrfpp_pol_to_csb, bleu_csb_to_pol, chrfpp_csb_to_pol
