# import libraries
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from typing import List
import torch

# Load tokenizer and model from HuggingFace
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

# Move model to GPU if available - setup device agnostic code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


## function to create embeddings
def create_embeddings(texts: List[str], batch_size: int):
    all_embeddings = []
    print(f"Total number of records: {len(texts)}")
    print(f"Num batches: {(len(texts) // batch_size) + 1}")

    # Extract embeddings for the texts in batches
    for start_index in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[start_index:start_index + batch_size]

        # Generate tokens and move input tensors to GPU
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Extract the embeddings. no_grad because the gradient does not need to be computed
        # since this is not a learning task
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the last hidden stated and pool them into a mean vector calculated across the sequence length dimension
        # This will reduce the output vector from [batch_size, sequence_length, hidden_layer_size]
        # to [batch_size, hidden_layer_size] thereby generating the embeddings for all the sequences in the batch
        last_hidden_states = outputs.last_hidden_state
        embeddings = torch.mean(last_hidden_states, dim=1).cpu().tolist()

        # Append to the embeddings list
        all_embeddings.extend(embeddings)

    return all_embeddings
	
	
# Create embeddings for the training and test set
## train set embeddings
train_embeddings = create_embeddings(texts=X_train["cleaned_text"].tolist(), batch_size=256)
train_embeddings_df = pd.DataFrame(train_embeddings)

## test set embeddings
test_embeddings = create_embeddings(texts=X_test["cleaned_text"].tolist(), batch_size=256)
test_embeddings_df = pd.DataFrame(test_embeddings)
