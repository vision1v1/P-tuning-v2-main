import os
import torch
import numpy as np
from tokenization_bert import BertTokenizer
from modeling_bert import BertModel, BertForMaskedLM
from configuration_bert import BertConfig

torch.set_printoptions(linewidth=500)

bert_large_uncased_path = os.path.join(os.getenv("my_data_dir"), "pretrained", "bert-large-cased")
bert_base_cased_path = os.path.join(os.getenv("my_data_dir"), "pretrained", "bert-base-cased")

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

def debug_pipeline():
    """调试pipeline分析原理"""
    from transformers.models.bert.modeling_bert import BertForMaskedLM # 定位源码
    from transformers import pipeline
    from transformers.pipelines.fill_mask import FillMaskPipeline
    unmasker:FillMaskPipeline = pipeline('fill-mask', model=bert_base_cased_path)
    outputs = unmasker("The man worked as a [MASK].")
    print("outputs =")
    for output in outputs:
        print(output, end='\n\n')

def debug_bert_for_masked_lm():
    """
    主要调试 BertForMaskedLM
    """

    tokenizer = BertTokenizer.from_pretrained(bert_base_cased_path)
    text = "The man worked as a [MASK]."
    
    # preprocess
    model_inputs = tokenizer(text, return_tensors='pt')

    model = BertForMaskedLM.from_pretrained(bert_base_cased_path)
    model_outputs = model.forward(**model_inputs)

    # postprocess
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


if __name__ == "__main__":
    # debug_tokenizer()
    # debug_model_inputs()
    # debug_config()
    # debug_model()
    # debug_pipeline()
    debug_bert_for_masked_lm()
    ...