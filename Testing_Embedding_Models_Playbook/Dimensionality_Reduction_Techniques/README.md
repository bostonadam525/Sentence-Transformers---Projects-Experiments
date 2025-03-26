# Dimensionality Reduction Techniques
* These are embedding evaluation methods based on the unsupervised learning methods of dimensionality reduction.

## Examples
1. The notebook "Cross Encoder Rerank" while it focuses on advanced retrieval for RAG systems from a vector DB using a cross encoder, prior to implementing the cross encoder I use UMAP to reduce the embeddings to 2-D and this allows us to see 2 things:
   * a. Which dataset in the vector database the query went to. As an example, a "food related query" goes to the "restaurant dataset" not the bank dataset.
   * b. The top_k (e.g. 6) most related documents or neighbors to that query embedding.
   * c. This technique allows you to evaluate queries in relation to the vectors in your vector database and if you need to consider using techniques such as query expansion reranking, data augmentation and more. 
