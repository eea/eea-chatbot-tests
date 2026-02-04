import os
import numpy as np
import transformers
import logging

# Suppress all warnings and progress bars
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


transformers.logging.set_verbosity_error()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("all-MiniLM-L6-v2")


def has_marker(data, *markers):
    """Check if test data has any of the specified markers."""
    return any(m in data.get("markers", []) for m in markers)


def quality_check_stages(score):
    """Return the quality check stage for a given score."""
    stages = [
        [0, 19, "Not supported - likely guesses, always double-check"],
        [20, 39, "Mostly not supported based on AI logic - please verify elsewhere"],
        [40, 79, "Partially supported - double-check if using for important decisions"],
        [80, 94, "Mostly supported - consider checking key points"],
        [95, 100, "Fully supported - safe to trust, no need to double-check"]
    ]

    for [low, high, message] in stages:
        if score >= low and score <= high:
            return message

    raise ValueError(f"Invalid quality check score: {score}% - expected between 0% and 100%")


# def embed(text):
#     """Encode a single text or list of texts into embeddings."""
#     return model.encode(text, normalize_embeddings=True)


# def cosine_similarity(text1, text2):
#     """Compute cosine similarity between two texts efficiently."""
#     # Encode both texts in a single batch for efficiency
#     embeddings = model.encode([text1, text2], normalize_embeddings=True)
#     return np.dot(embeddings[0], embeddings[1])
