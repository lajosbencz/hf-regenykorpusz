import os
import pandas as pd
from lxml import etree
from datasets import Dataset, Features, Value
from datasets.utils.logging import disable_progress_bar
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
SHARD_SIZE = os.getenv("SHARD_SIZE", "10MB")
LOCAL_OUTPUT_DIR = os.getenv("LOCAL_OUTPUT_DIR", "./parquet")

df_meta = pd.read_csv(METADATA_PATH, sep="\t")
meta_map = df_meta.set_index("novel_id").to_dict("index")


# Define Features (Essential for Dataset Viewer and Parquet typing)
features = Features(
    {
        "text": Value("string"),
        "paragraph_index": Value("int32"),
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
        paragraph_index = -1

        for _, elem in context:
            text = "".join(elem.itertext()).strip()
            if text:
                paragraph_index += 1
                yield {
                    "text": text,
                    "paragraph_index": paragraph_index,
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
    disable_progress_bar()
    
    if not LOCAL_ONLY:
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN must be set.")
        login(token=HF_TOKEN)

    print("Step 1/2: Initializing generator...")
    ds = Dataset.from_generator(
        paragraph_generator,
        gen_kwargs={"level1_path": LEVEL1_DIR, "meta_dict": meta_map},
        features=features,
    )
    
    print("Step 1.5: Sorting and flattening (this ensures sequential streaming)...")
    ds = ds.sort(["author", "title", "paragraph_index"], keep_in_memory=False).flatten_indices()

    if LOCAL_ONLY:
        print(f"Step 2/2: Writing to local Parquet files with {SHARD_SIZE} shards...")
        os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
        ds.save_to_disk(
            LOCAL_OUTPUT_DIR, 
            max_shard_size=SHARD_SIZE,
            num_proc=2
        )
    else:
        print(f"Step 2/2: Uploading to Hub with {SHARD_SIZE} shards...")
        ds.push_to_hub(
            HF_REPO_ID, 
            max_shard_size=SHARD_SIZE, 
            private=False
        )

    print("Process complete!")
