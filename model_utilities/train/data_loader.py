import os
import itertools

import pandas as pd
import datasets as ds


def load_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path, sep="\t", index_col=0)


def load_dataset(train_path: str, validation_path: str) -> ds.DatasetDict:
    paths = {
        "train": train_path,
        "validation": validation_path,
    }

    dataset = ds.load_dataset("csv", data_files=paths, sep="\t")

    # Removing the incremental id column as we dont need it
    dataset = dataset.remove_columns("Unnamed: 0")

    return dataset


def prepare_train_dataset(data_directory: str, weights: dict, seed: int) -> ds.DatasetDict:
    paths = {key: {word: os.path.join(data_directory, f"{word}-{key}.tsv") for word in ["train", "val"]} for key in weights.keys()}

    datasets = {key: load_dataset(paths[key]["train"], paths[key]["val"]) for key in paths}

    # HF datasets do not support interleaving a DatasetDict
    # so we need to interleave each split separately
    train_datasets = {key: datasets[key]["train"] for key in datasets}

    train_dataset_combined = ds.interleave_datasets(list(train_datasets.values()),
                                                    list(weights.values()),
                                                    seed=seed,
                                                    stopping_strategy="all_exhausted",
                                                    )
    validation_datasets = {key: datasets[key]["validation"] for key in datasets}

    validation_dataset_combined = ds.interleave_datasets(list(validation_datasets.values()),
                                                         list(weights.values()),
                                                         seed=seed,
                                                         stopping_strategy="all_exhausted",
                                                         )
    final_dataset = ds.DatasetDict({
        "train": train_dataset_combined,
        "validation": validation_dataset_combined
    })

    return final_dataset


def load_synonyms(data_path: str) -> list[list[str]]:
    references = []

    with open(data_path) as file:
        for line in file.readlines():
            line = line[0:-1]
            if line.startswith("ZAPIS SYMBOLICZNY"):
                references.append([])
            elif len(line) != 0:
                references[-1].append(line)

    # print(references)

    # We need to transpose the references "matrix" so that each row contains one sentence matching each input sentence
    # (The code above has each row containing references to only one sentence)
    references_transposed = list(map(list, itertools.zip_longest(*references, fillvalue=None)))
    return references_transposed
