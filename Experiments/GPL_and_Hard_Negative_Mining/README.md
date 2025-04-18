# Generative Pseudo Labeling (GPL) and Hard Negative Mining
* A repo devoted to these very important concepts that came out of the UKP lab from sentence transformers.


# What is GPL?
* Generative Pseudo Labeling (GPL) is a technique for domain adaptation in dense retrieval, using a cross-encoder to generate pseudo-relevance labels for queries and documents.

---
# How do we use GPL?
1. **Fine tune a pre-trained model**
   * "Pre-trained" is a transformers model that has NOT been pretrained already for semantic search.
   * Example: Take `bert-base` model and fine-tune it for semantic search.
     * The results would probably be "satisfactory" as `bert-base` is not the most ideal model for semantic search adaptation.

 2. **Domain Adapation**
  * You already have a semantic search pre-trained model.
  * As an example, lets say the domain is very specific such as **COVID-19** but the model is high dimensional and trained on general medical data so it may not be able to recognize, parse, or infer the specific terminology such as acronyms, abbreviations and colloquial language. 

---
## Why do we need GPL?
* Large pre-trained embedding models perform well in high dimensional spaces but often perform **poorly when shifted to domain specific data** such as COVID-19 or other domains that have specific terminology, acronyms, and abbreviations.
* Thus, there is a need to create a method that can effectively **adapt dense retrieval models to new domains without requiring large amounts of labeled data.** 
  * GPL (Generative Pseudo Labeling), is one such method.
  * GPL is an **unsupervised domain adaptation technique** for dense retrieval models that combines a query generator with pseudo-labeling.
  * GPL uses a T5 model to generate queries for a target domain.
  * It retrieves negative passages using an existing dense retrieval model and uses a cross-encoder to score (query, passage) pairs.
* GPL outperforms other domain adaptation methods.
  * It improves performance by up to 9.3 nDCG@10 (normalized Discounted Cumulative Gain at rank 10) over models trained on MS MARCO and up to 4.5 nDCG@10 over QGen (Query generation). [source](https://zilliz.com/blog/generative-pseudo-labeling-for-unsupervised-domain-adaptation-of-dense-retrieval)

![image](https://github.com/user-attachments/assets/0b4c1f8f-2171-4f7d-bd26-fd260456f021)

### Examples of use cases for GPL
* There are numerous use cases where you may have large amounts of unlabeled data and need to generate labels for this data.
  * Example 1: Text scraped from web pages
  * Example 2: PDF documents
  * Example 3: Medical Records
  * Example 4: Financial Records
  * ...etc...

## What is the goal of GPL?
* The goal is to adapt a dense retrieval model (which maps queries and documents to vectors) to a new domain or dataset where labeled data is scarce. 

## Training Process for GPL
* Each step of the process requires a pre-trained model.
  1. **Query Generation**
     * A model (like a T5 encoder-decoder) generates synthetic queries for each document in the target domain. 
  2. **Negative Mining**
     * A pre-trained dense retrieval model is used to find **similar** (but not relevant) documents (negative passages) for each generated query. 
  3. **Pseudo Labeling**
     * A cross-encoder model is used to assign relevance scores to the query-document pairs, creating pseudo-labels that represent the model's confidence in the relevance of a query-document pair. 
  4. **Fine-tuning**
     * The dense retrieval model is then fine-tuned using the generated query-document pairs and their pseudo-labels, improving its ability to retrieve relevant documents in the target domain. 

* **Benefits**
  * GPL can be used to adapt dense retrieval models to new domains with minimal labeled data, making it an efficient unsupervised domain adaptation technique. 


# What is Hard Negative Mining?
* Hard negative mining focuses on selecting difficult negative examples (those that are close to positives but shouldn't be) to improve model performance.

## What is the goal of Hard Negative Mining?
* The goal is to improve the performance of classification or ranking models by focusing on difficult negative examples during training. 
* **How it works**
  1. **Identify Hard Negatives**
     * The model identifies negative examples that are close to positive examples in the feature space, making them difficult to distinguish. 

  2. **Include in Training**
     * These hard negative examples are then included in the training data, forcing the model to learn more discriminative features. 

* **Benefits**
  * By focusing on hard negatives, the model learns to distinguish between similar but different classes or items, leading to better performance. 
* **Example**
  * In object detection, hard negative mining might involve selecting bounding boxes that are similar to the target object but are not the correct object. 

