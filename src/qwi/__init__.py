"""Local text-to-image pipeline for Apple Silicon."""

import os

# The runtime never downloads (PLAN.md §5.7): huggingface_hub reads this once at import, so set it
# before any qwi module can import it. Forced, not defaulted, so a stray shell value can't
# start a multi-GB download mid-run.
os.environ["HF_HUB_OFFLINE"] = "1"
