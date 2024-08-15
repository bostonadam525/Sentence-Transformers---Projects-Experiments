## Creating embeddings using Tensorflow
## makes sure that you have already run: !pip install sentence_transformers
from sentence_transformers import SentenceTransformer
import tensorflow as tf

## function for creating embeddings
%%time

def embed(model, model_type, sentences):
  if model_type == 'use':
    embeddings = model(sentences)
  elif model_type == 'sentence transformer':
    embeddings = model.encode(sentences)

  return embeddings


## instantiate SentenceTransformer
model_st1 = SentenceTransformer('all-mpnet-base-v2', device='cuda') ## set to cpu or cuda

## create embeddings
embeddings_st1 = embed(model_st1, 'sentence transformer', all_intents)


## check shape of embeddings
embeddings_st1.shape
