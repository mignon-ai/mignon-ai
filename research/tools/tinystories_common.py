"""Shared by the TinyStories tools: the dataset and the document separator mignon expects."""

from datasets import load_dataset

DATASET = "roneneldan/TinyStories"
# Matches data.DocSeparator in internal/data: ends a document in a corpus file.
DOC_SEPARATOR = "\n\n<EOS>\n\n"


def load_stories(split, n_stories=None):
    """Stream stories from a split ('train' or 'validation'); all of them when n_stories is None."""
    ds = load_dataset(DATASET, split=split, streaming=True)
    for i, ex in enumerate(ds):
        if n_stories is not None and i >= n_stories:
            return
        yield ex["text"]
