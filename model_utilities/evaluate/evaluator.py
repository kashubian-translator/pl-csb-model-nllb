from logging import Logger

import pandas as pd
import sacrebleu
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

from translate.translator import Translator


class ModelEvaluator:
    __logger: Logger

    def __init__(self, logger, model: AutoModelForSeq2SeqLM, tokenizer: NllbTokenizer) -> None:
        self.__logger = logger
        self.__translator = Translator(self.__logger, model, tokenizer)

    def evaluate(self, sentences: pd.Series, references: pd.Series) -> tuple[sacrebleu.metrics.BLEUScore, sacrebleu.metrics.CHRFScore]:
        source_lang = sentences.name
        target_lang = references.name

        translated_sentences_list = self.__translator.translate(sentences.to_list(), source_lang, target_lang)

        # There can be multiple references per sentence so we need to pass a one element list
        references_list = [references.to_list()]

        bleu = sacrebleu.corpus_bleu(translated_sentences_list, references_list)
        chrfpp = sacrebleu.corpus_chrf(translated_sentences_list, references_list, word_order=2)

        return (bleu, chrfpp)

    def evaluate_with_synonyms(self, sentences: list[str], references: list[list[str]], source_lang: str, target_lang: str) -> tuple[sacrebleu.metrics.BLEUScore, sacrebleu.metrics.CHRFScore]:
        translated_sentences = self.__translator.translate(sentences, source_lang, target_lang)

        bleu = sacrebleu.corpus_bleu(translated_sentences, references)
        chrfpp = sacrebleu.corpus_chrf(translated_sentences, references, word_order=2)

        return (bleu, chrfpp)
