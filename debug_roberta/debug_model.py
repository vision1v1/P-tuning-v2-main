import os
import torch
import numpy as np
from tokenization_roberta import RobertaTokenizer
from modeling_roberta import RobertaModel, RobertaForMaskedLM
from configuration_roberta import RobertaConfig
from transformers.tokenization_utils_base import BatchEncoding

roberta_large_path = os.path.join(os.getenv("my_data_dir"), "pretrained", "roberta-large")
roberta_base_path = os.path.join(os.getenv("my_data_dir"), "pretrained", "roberta-base")

def debug_tokenizer():
    """
    调试分词
    """
    tokenizer = RobertaTokenizer.from_pretrained(roberta_large_path)
    text = "Replace me by any text you'd like."
    tokens = tokenizer.tokenize(text)
    print("text =", text, "tokens =", tokens, sep='\n', end='\n\n')

def debug_model_inputs():
    """
    调试模型输入
    """
    tokenizer = RobertaTokenizer.from_pretrained(roberta_large_path)
    text = "Replace me by any text you'd like."
    encoded_input:BatchEncoding = tokenizer(text, return_tensors='pt')
    print("encode_input =", type(encoded_input), sep='\n', end='\n\n')

    for key, value in encoded_input.items():
        print(key, value, sep='\n', end="\n\n")

def debug_model_config():
    """
    调试模型配置
    """
    config = RobertaConfig.from_pretrained(roberta_large_path)
    print("config", config, sep='\n', end='\n\n')
    

def debug_model_forward():
    """
    调试模型 forward
    """
    config = RobertaConfig.from_pretrained(roberta_large_path)
    config.num_hidden_layers = 2 # 方便调试

    model = RobertaModel(config=config) # 方便调试
    
    # model = RobertaModel.from_pretrained(roberta_large_path)

    tokenizer = RobertaTokenizer.from_pretrained(roberta_large_path)
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')

    # forward
    output = model.forward(**encoded_input)

    print("output =", type(output), sep='\n', end='\n\n')

    for key, value in output.items():
        print(key, value, sep='\n', end='\n\n')


def debug_fill_mask():
    """
    用 RoBERTa 调试 fill mask
    """
    from transformers import pipeline
    unmasker = pipeline('fill-mask', model=roberta_base_path)
    output = unmasker("Hello I'm a <mask> model.")
    print("output =", type(output), end='\n\n')
    for o in output:
        print(o, end='\n\n')

def debug_roberta_fill_mask():
    
    tokenizer = RobertaTokenizer.from_pretrained(roberta_base_path)
    model = RobertaForMaskedLM.from_pretrained(roberta_base_path)
    
    text = "Hello I'm a <mask> model."
    model_inputs:BatchEncoding = tokenizer(text, return_tensors='pt')
    model_outputs = model.forward(**model_inputs)

    model_outputs["input_ids"] = model_inputs["input_ids"] # 源码直接就这样复制的
    input_ids = model_outputs["input_ids"][0]
    outputs = model_outputs["logits"]
    masked_index = torch.nonzero(input_ids == tokenizer.mask_token_id, as_tuple=False).squeeze(-1)
    logits = outputs[0, masked_index, :]
    probs = logits.softmax(dim=-1)

    top_k = 5 # 
    values, predictions = probs.topk(top_k)

    result = []
    single_mask = values.shape[0] == 1
    for i, (_values, _predictions) in enumerate(zip(values.tolist(), predictions.tolist())):
        row = []
        for v, p in zip(_values, _predictions):
            # Copy is important since we're going to modify this array in place
            tokens = input_ids.numpy().copy()

            tokens[masked_index[i]] = p
            # Filter padding out:
            tokens = tokens[np.where(tokens != tokenizer.pad_token_id)]
            # Originally we skip special tokens to give readable output.
            # For multi masks though, the other [MASK] would be removed otherwise
            # making the output look odd, so we add them back
            sequence = tokenizer.decode(tokens, skip_special_tokens=single_mask)
            proposition = {"score": v, "token": p, "token_str": tokenizer.decode([p]), "sequence": sequence}
            row.append(proposition)
        result.append(row)

    if single_mask:
        result = result[0]
    
    print("result =")
    for r in result:
        print(r, end='\n\n')
    ...


if __name__ == "__main__":
    # debug_tokenizer()
    # debug_model_inputs()
    # debug_model_config()
    # debug_model_forward()
    # debug_fill_mask()
    debug_roberta_fill_mask()
    ...