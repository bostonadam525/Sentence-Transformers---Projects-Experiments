## scripts for batching embeddings with huggingface accelerator
## Script #1 -- using open ai embeddings

from tqdm import tqdm
from typing import List, Callable
from langchain.embeddings import OpenAIEmbeddings
#from langchain.docstore.document import Document
from accelerate import Accelerator

## create embeddings in batches
def embed_documents_in_batches(
    docs: List[Document],
    embeddings_model: Callable[[List[str]], List[List[float]]],
    batch_size: int = 256,
) -> List[List[float]]:
    """
    Embeds documents in batches using the provided embeddings model and potentially using 
    the 'accelerate' library for parallelization.

    Args:
        docs: A list of Document objects to embed.
        embeddings_model: The embeddings model to use (e.g., OpenAIEmbeddings).
        batch_size: The size of each batch of documents to embed.

    Returns:
        A list of embeddings, where each embedding corresponds to a document.
    """

    accelerator = Accelerator() 
    # Use accelerator.device instead of accelerator.to_device()
    embeddings_model = accelerator.prepare(embeddings_model)  # Prepare the model

    all_embeddings = []
    for i in tqdm(range(0, len(docs), batch_size), desc="Embedding documents"):
        batch = docs[i : i + batch_size]
        # Instead of using 'with accelerator.device_placement():', move the model and data to the device
        # before the loop if needed:
        # embeddings_model.to(accelerator.device) 
        embeddings = embeddings_model([d.page_content for d in batch])  
        all_embeddings.extend(embeddings)
    return all_embeddings

# apply function to docs
embeddings = embed_documents_in_batches(docs, openai_embed_model.embed_documents)





## script #2 using huggingface embedding model
%%time
#!pip install accelerate
import torch
from tqdm.auto import tqdm
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from accelerate import Accelerator

# Function to embed documents in batches
def embed_documents_in_batches(docs, model, tokenizer, batch_size=256):
    accelerator = Accelerator()
    model, tokenizer, docs = accelerator.prepare(model, tokenizer, docs)  # Prepare with accelerator
    
    all_embeddings = []
    for i in tqdm(range(0, len(docs), batch_size), desc="Embedding documents", disable=not accelerator.is_local_main_process): 
        batch = docs[i : i + batch_size]

        # Tokenize the batch of documents
        with torch.no_grad():  # Disable gradient calculation during inference
            inputs = tokenizer([d.page_content for d in batch], padding=True, truncation=True, return_tensors="pt").to(accelerator.device) 
            embeddings = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()  # Extract embeddings and move to CPU
        
        all_embeddings.extend(embeddings)

    # Gather embeddings from all processes if using multi-GPU or distributed setup
    all_embeddings = accelerator.gather(all_embeddings) 
    
    return all_embeddings

# ... (Rest of your code to load model, tokenizer, and documents)

# Example usage:
model_name = "mixedbread-ai/mxbai-embed-large-v1"  # Or your desired model
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

## apply function to create embeddings
embeddings = embed_documents_in_batches(docs, model, tokenizer)
