from utils.setup import setup_env
from huggingface_hub import list_models

setup_env()


# see openlm_models.txt for all models
models_3b = [
    "openlm-research/open_llama_3b",
    "openlm-research/open_llama_3b_easylm",
    "openlm-research/open_llama_3b_step_2500",
    "openlm-research/open_llama_3b_step_5000",
    "openlm-research/open_llama_3b_step_7500",
    "openlm-research/open_llama_3b_step_10000",
    "openlm-research/open_llama_3b_step_50000",
    "openlm-research/open_llama_3b_step_100000",
    "openlm-research/open_llama_3b_step_150000",
    "openlm-research/open_llama_3b_step_200000",
    "openlm-research/open_llama_3b_step_250000",
]

models_7b = [
    "openlm-research/open_llama_7b",
    "openlm-research/open_llama_7b_easylm",
    "openlm-research/open_llama_7b_step_2500",
    "openlm-research/open_llama_7b_step_5000",
    "openlm-research/open_llama_7b_step_7500",
    "openlm-research/open_llama_7b_step_10000",
    "openlm-research/open_llama_7b_step_50000",
    "openlm-research/open_llama_7b_step_100000",
    "openlm-research/open_llama_7b_step_150000",
    "openlm-research/open_llama_7b_step_200000",
    "openlm-research/open_llama_7b_step_250000",
]

models_13b = [
    "openlm-research/open_llama_13b",
    "openlm-research/open_llama_13b_easylm",
    "openlm-research/open_llama_13b_step_5000",
    "openlm-research/open_llama_13b_step_10000",
    "openlm-research/open_llama_13b_step_15000",
    "openlm-research/open_llama_13b_step_20000",
    "openlm-research/open_llama_13b_step_100000",
    "openlm-research/open_llama_13b_step_200000",
    "openlm-research/open_llama_13b_step_300000",
    "openlm-research/open_llama_13b_step_400000",
    "openlm-research/open_llama_13b_step_500000",
]

# now get pretraining loss for all models

def get_pretraining_loss(model):
    # set up a script that gets the loss of a model on some small(ish) pretraining corpus val set
    # To compare them, we use held-out loss on the C4 validation set as our independent variable L(M)
    # TODO
    pass

pretraining_losses = [get_pretraining_loss(model) for model in models_3b]
# TODO write to file