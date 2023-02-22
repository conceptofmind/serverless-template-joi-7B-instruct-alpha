# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A HuggingFace model is downloaded

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    joi_map= {
        'gpt_neox.embed_in': 0,
        'gpt_neox.layers': 0,
        'gpt_neox.final_layer_norm': 0,
        'embed_out': 0
    }
    name = "Rallio67/joi_7B_instruct_alpha"
    print("downloading model...")
    AutoModelForCausalLM.from_pretrained(
        name, 
        device_map=joi_map, 
        torch_dtype=torch.float16,
        )
    print("done")
    print("downloading tokenizer...")
    AutoTokenizer.from_pretrained(name)
    print("done")
    
if __name__ == "__main__":
    download_model()