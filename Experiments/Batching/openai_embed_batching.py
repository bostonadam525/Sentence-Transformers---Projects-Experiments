## scripts for batching openai embeddings
## script #1

#%%time
from tqdm.auto import tqdm
import pandas as pd # Import pandas

# Function to embed documents in batches
def embed_documents_in_batches(docs, embeddings_model, batch_size=256):
    all_embeddings = []
    for i in tqdm(range(0, len(docs), batch_size), desc="Embedding documents"): # Use tqdm for progress bar
        batch = docs[i : i + batch_size]
        ## if using openai embeddings run this code below
        embeddings = embeddings_model.embed_documents([d.page_content for d in batch])
        all_embeddings.extend(embeddings)
    return all_embeddings

# Get the embeddings -- make sure to use embedding model you are using
embeddings = embed_documents_in_batches(docs, openai_embed_model) # Call the function directly




