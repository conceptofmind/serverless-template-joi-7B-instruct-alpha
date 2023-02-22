from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global tokenizer

    name = "Rallio67/joi_7B_instruct_alpha"
    joi_map= {'gpt_neox.embed_in': 0,
        'gpt_neox.layers': 0,
        'gpt_neox.final_layer_norm': 0,
        'embed_out': 0}
    model = AutoModelForCausalLM.from_pretrained(
        name, 
        device_map=joi_map,
        torch_dtype=torch.float16,
        load_in_8bit=True
        )
    tokenizer = AutoTokenizer.from_pretrained(name)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    encoded_input = tokenizer(prompt, return_tensors='pt')
    output_sequences = model.generate(
        input_ids=encoded_input['input_ids'].cuda(0),
        do_sample=True,
        max_new_tokens=35,
        num_return_sequences=1,
        top_p=0.95,
        temperature=0.5,
        penalty_alpha=0.6,
        top_k=4,
        output_scores=True,
        return_dict_in_generate=True,
        repetition_penalty=1.03,
        eos_token_id=0,
        use_cache=True
        )
    gen_sequences = output_sequences.sequences[:, encoded_input['input_ids'].shape[-1]:]
    for sequence in gen_sequences:
        new_line=tokenizer.decode(sequence, skip_special_tokens=True)
    result = {"output": new_line}
    return result

if __name__ == "__main__":
    init()
    model_inputs = {'prompt': 'Hello, my dog is cute? \n\nJoi:'}
    print(inference(model_inputs))