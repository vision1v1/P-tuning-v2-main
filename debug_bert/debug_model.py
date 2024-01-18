import os
import torch
from tokenization_bert import BertTokenizer
from modeling_bert import BertModel
from configuration_bert import BertConfig

torch.set_printoptions(linewidth=500)

bert_large_uncased_path = os.path.join(os.getenv("my_data_dir"), "pretrained", "bert-large-cased")

def debug_tokenizer():
    """
    调试分词
    """
    tokenizer = BertTokenizer.from_pretrained(bert_large_uncased_path)
    text = "Replace me by any text you'd like."
    print("text =", text, sep='\n', end='\n\n')

    # 分词统一接口
    tokens = tokenizer.tokenize(text)
    print("tokens =", tokens, sep='\n', end='\n\n')


    # encoded_input = tokenizer(text, return_tensors='pt')
    # print(encoded_input)


def debug_model_inputs():
    """
    调试模型输入
    """
    tokenizer = BertTokenizer.from_pretrained(bert_large_uncased_path)
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    
    for key, value in encoded_input.items():
        print(key, value, sep='\n', end='\n\n')
    

def debug_config():
    """
    调试模型 配置
    """
    config = BertConfig.from_pretrained(bert_large_uncased_path)
    print("config =", config, sep='\n', end='\n\n')


def debug_model():
    """
    方便学习
    """
    
    config = BertConfig.from_pretrained(bert_large_uncased_path)
    config.num_hidden_layers = 2 # 方便调试

    model = BertModel(config=config) # 方便调试
    # model = BertModel.from_pretrained(bert_large_uncased_path)

    tokenizer = BertTokenizer.from_pretrained(bert_large_uncased_path)
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    
    # forward
    output = model.forward(**encoded_input)

    print(type(output), end='\n\n')
    for key, value in output.items():
        print(key, value, sep='\n', end='\n\n')


if __name__ == "__main__":
    # debug_tokenizer()
    # debug_model_inputs()
    # debug_config()
    debug_model()
    ...