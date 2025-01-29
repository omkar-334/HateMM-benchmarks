import json
import re

import torch
from transformers import BertModel, BertTokenizer


def clean_str(s: str) -> str:
    """Cleans a string for tokenization."""
    s = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", s)
    s = re.sub(r"\'s", " 's", s)
    s = re.sub(r"\'ve", " 've", s)
    s = re.sub(r"n\'t", " n't", s)
    s = re.sub(r"\'re", " 're", s)
    s = re.sub(r"\'d", " 'd", s)
    s = re.sub(r"\'ll", " 'll", s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\"", ' " ', s)
    s = re.sub(r"\(", " ( ", s)
    s = re.sub(r"\)", " ) ", s)
    s = re.sub(r"\?", " ? ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"\.", " . ", s)
    s = re.sub(r"., ", " , ", s)
    s = re.sub(r"\\n", " ", s)
    return s.strip().lower()


def extract_bert_features(sentences, output_file, model_name="bert-base-uncased", layers=[-1, -2, -3, -4]):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()

    results = []

    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)

        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.hidden_states
        selected_features = [hidden_states[layer_idx].squeeze(0) for layer_idx in layers]

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        features_dict = {
            "index": len(results),
            "features": [
                {"token": token, "layers": [{"index": layer_idx, "values": feature[token_idx].tolist()} for layer_idx, feature in zip(layers, selected_features)]}
                for token_idx, token in enumerate(tokens)
            ],
        }

        results.append(features_dict)

    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
