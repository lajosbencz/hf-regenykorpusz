import os
import pandas as pd
from lxml import etree
from datasets import Dataset, Features, Value
from huggingface_hub import login

# Set the environment variable WRITE_LOCAL=0 to upload to Hugging Face.
LOCAL_ONLY_ENV = os.getenv("WRITE_LOCAL", "1")
LOCAL_ONLY = LOCAL_ONLY_ENV == "1" or LOCAL_ONLY_ENV.lower() == "true"

# https://github.com/ELTE-DH/regenykorpusz
SOURCE_DIR = os.getenv("SOURCE_DIR", "regenykorpusz")
LEVEL1_DIR = os.path.join(SOURCE_DIR, "level1")
METADATA_PATH = os.path.join(SOURCE_DIR, "level1_metadata.tsv")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID", "lazos/literature-hu-test")
HF_SPLIT_TEST_SIZE = float(os.getenv("HF_SPLIT_TEST_SIZE", "0.05"))
HF_SPLIT_SEED = int(os.getenv("HF_SPLIT_SEED", "42"))
LOCAL_OUTPUT_DIR = os.getenv("LOCAL_OUTPUT_DIR", "./parquet")

df_meta = pd.read_csv(METADATA_PATH, sep="\t")
meta_map = df_meta.set_index("novel_id").to_dict("index")


# Define Features (Essential for Dataset Viewer and Parquet typing)
features = Features(
    {
        "text": Value("string"),
        "author": Value("string"),
        "title": Value("string"),
        "first_edition": Value("string"),
        "num_word": Value("int32"),
        "size": Value("string"),
        "canonicity": Value("string"),
        "author_gender": Value("string"),
        "novel_id": Value("string"),
    }
)


# Define the Streaming Generator
def paragraph_generator(level1_path, meta_dict):
    """Memory-efficient paragraph streaming from TEI XML."""
    for novel_id, novel_meta in meta_dict.items():
        filename = f"{novel_id}.xml"
        file_path = os.path.join(level1_path, filename)

        if not os.path.exists(file_path):
            continue

        # Use iterparse for event-driven parsing to prevent memory buildup
        context = etree.iterparse(file_path, events=("end",), tag="{*}p")

        for _, elem in context:
            text = "".join(elem.itertext()).strip()
            if text:
                yield {
                    "text": text,
                    "author": str(novel_meta.get("author_name", "Unknown")),
                    "title": str(novel_meta.get("title", "Unknown")),
                    "first_edition": str(novel_meta.get("first_edition", "Unknown")),
                    "num_word": int(novel_meta.get("num_word", 0)),
                    "size": str(novel_meta.get("size", "Unknown")),
                    "canonicity": str(novel_meta.get("canonicity", "Unknown")),
                    "author_gender": str(novel_meta.get("author_gender", "Unknown")),
                    "novel_id": novel_id,
                }

            # CRITICAL: Clear element and siblings to free RAM
            elem.clear()
        del context


if __name__ == "__main__":

    # Authenticate only if we are uploading to the Hub
    if not LOCAL_ONLY:
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN environment variable is not set. Please set the HF_TOKEN secret in your repository settings.")
        login(token=HF_TOKEN)

    # Create and Split Dataset
    print("Step 1/2: Initializing generator and splitting data...")
    # This pull records by yielding them from the generator
    ds = Dataset.from_generator(
        paragraph_generator,
        gen_kwargs={"level1_path": LEVEL1_DIR, "meta_dict": meta_map},
        features=features,
    )

    ds_dict = ds.train_test_split(test_size=HF_SPLIT_TEST_SIZE, seed=HF_SPLIT_SEED)

    # Execution: Local Save or Hub Push
    if LOCAL_ONLY:
        print(f"Step 2/2: Writing to local Parquet files in {LOCAL_OUTPUT_DIR}...")
        os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
        # Directly save to parquet to save space/time
        for split, dataset in ds_dict.items():
            output_path = os.path.join(LOCAL_OUTPUT_DIR, f"{split}.parquet")
            dataset.to_parquet(output_path)
            print(f"Saved {split} split to {output_path}")
    else:
        print(f"Step 2/2: Streaming upload to Hub: {HF_REPO_ID}...")
        ds_dict.push_to_hub(HF_REPO_ID, max_shard_size="50MB", private=False)

    print("Process complete!")
