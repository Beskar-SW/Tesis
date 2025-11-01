import numpy as np
from transformers import AutoTokenizer, AutoModel
from bpe import BPE
from w2v import Word2VecSubword
from test import WordEmbedder
import torch

# ====================================================
# 1️⃣ Tu modelo personalizado (BPE + Word2Vec)
# ====================================================
tokenizer = BPE()
tokenizer.load("bpe_tokenizer.json")

word2vec_model = Word2VecSubword(tokenizer)
word2vec_model.load_model("subword_word2vec.model")

embedder = WordEmbedder(tokenizer, word2vec_model)

pair = ("palafox", "pallafox")

custom_sim = embedder.get_word_similarity(*pair)
print("🧩 SIMILITUD (BPE+Word2Vec):")
print(f"  {pair[0]} vs {pair[1]} → {custom_sim:.4f}" if custom_sim else "  ❌ No se pudo calcular")


# ====================================================
# 2️⃣ Modelo BERT Multilingüe (contextual)
# ====================================================
print("\n🤖 SIMILITUD (BERT-base-multilingual-cased):")

bert_name = "bert-base-multilingual-cased"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_name)
bert_model = AutoModel.from_pretrained(bert_name)

def bert_embed(text):
    inputs = bert_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Usamos el embedding de [CLS]
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

vec_a = bert_embed(pair[0])
vec_b = bert_embed(pair[1])
bert_sim = cosine(vec_a, vec_b)

print(f"  {pair[0]} vs {pair[1]} → {bert_sim:.4f}")
