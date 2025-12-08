# test_batch.py
import logging
import sys
import os

# Use package-relative imports
from .pipeline import batched_inference, inference_pdf, recursive_batched_inference


# 1. Define where your PDFs are
INPUT_DIR = "/home/crai/crai_projects/pure_visual_grounding/xpy"
try:
    result = recursive_batched_inference(INPUT_DIR)
except Exception as e:
    print(e)