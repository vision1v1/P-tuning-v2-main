import os
import torch
from tokenization_roberta import RobertaTokenizer
from modeling_roberta import RobertaModel
from configuration_roberta import RobertaConfig
from transformers.tokenization_utils_base import BatchEncoding

roberta_large_path = os.path.join(os.getenv("my_data_dir"), "pretrained", "roberta-large")

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


if __name__ == "__main__":
    # debug_tokenizer()
    # debug_model_inputs()
    # debug_model_config()
    debug_model_forward()
    ...