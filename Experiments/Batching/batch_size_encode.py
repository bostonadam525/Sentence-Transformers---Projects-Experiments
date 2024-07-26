#Batch Training Embeddings
## `batch_size` is a default parameter in the .encode() method in sentence transformers, default size is 32.
## This will only influence the speed of training the embeddings.


%%time
# embed all text in batches
text_chunk_embeddings = embedding_model.encode(text_chunks,
                                               batch_size=32, #default size
                                               convert_to_tensor=True) #convert to pt tensors

# output embeddings
text_chunk_embeddings






## another way to batch train embeddings in a list comprehension
## train embeddings in batches
%%time

text_chunks = [item['sentence_chunk'] for item in pages_and_chunks_over_min_token_len]
text_chunks[419]
