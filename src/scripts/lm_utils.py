from datasets import load_dataset

def load_data(train_data,val_data,extension):
    data_files         =  {}
    data_files["train"]      =  train_data
    data_files["validation"] =  val_data

    dataset = load_dataset(extension,data_files=data_files,
    cache_dir="huggingface",
    )
    return dataset


