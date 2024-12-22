import re
from logging import Logger

import torch
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM, Pipeline, pipeline


class Translator:
    __logger: Logger
    __model: AutoModelForSeq2SeqLM
    __tokenizer: NllbTokenizer
    __pipeline: Pipeline

    def __init__(self, logger: Logger, model: AutoModelForSeq2SeqLM, tokenizer: NllbTokenizer, max_length=200) -> None:
        self.__logger = logger

        self.__model = model

        self.__tokenizer = tokenizer
        self.__max_length = max_length

        device_index = 0 if torch.cuda.is_available() else -1

        self.__pipeline = pipeline(task="translation",
                                   model=self.__model,
                                   tokenizer=self.__tokenizer,
                                   max_length=self.__max_length,
                                   device=device_index)

    def translate(self, text_list: list[str], source_lang: str, target_lang: str) -> list[str]:
        model_output = self.__pipeline(text_list, src_lang=source_lang, tgt_lang=target_lang)

        translation_text_list = []
        for i, text in enumerate(text_list):
            translation_text = model_output[i]["translation_text"]
            translation_text_list.append(translation_text)
            self.__logger.debug(f"Translation: '{text}' ({source_lang}) → '{translation_text}' ({target_lang})")

        return translation_text_list
