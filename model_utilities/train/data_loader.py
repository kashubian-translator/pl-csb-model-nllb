import os

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
