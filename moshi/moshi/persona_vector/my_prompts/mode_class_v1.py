# Linear classifier for listening mode vs. speaking mode
# When 

class HiddenExtractor:
    def generate_batch(self, input_sentences: list[str], output_path: list[str]):
        """Generate the output_hidden for each input sentecnce use inference_batch to process all
        sentence in one session. and save as output_path. the outputp_path by default is output_hidden.pt
        now it should be the give name
        """
    
    def generate(self, input_sentence: str, output_path: str):
        """Generate the output_hidden for each input sentecnce"""

    def class_mode_dataset(self, dataset_path):
        """The dataset is at somewhere like Full-Duplex-Bench/data/mode_class/ and has 
        the format of dataset_path/*/input.json, where each input.json has two fields: 
        complete_sentence and incomplete_sentence (you can check the format yourself)
        for each input.json, generate two output_hidden.pt files: complete_sentence_hidden.pt and incomplete_sentence_hidden.pt
        """

class HiddenModeClassifier:
    def __init__(self):
        pass

    def train(self, dataset_path, output_dir, layer=-1):
        """The dataset is at somewhere like Full-Duplex-Bench/data/mode_class/ and has 
        the format of dataset_path/*/input.json, where each input.json has two fields: 
        complete_sentence and incomplete_sentence (you can check the format yourself)
        for each input.json, there are two output_hidden.pt files: complete_sentence_hidden.pt and incomplete_sentence_hidden.pt
        if no exist error quit.

        this is a simple linear classifier, that predict the mode (listening or speaking) for each token based on 
        hidden representation of layer `layer`. ignore the sentece text in input.json, you will see
        {
            ...
            "listening": [[4, 20], [31, 50], ...], # list of [start_token_id, end_token_id] for listening mode 
                                                    #[3,5] means 3,4,5 token are in listening mode, index starts from 0
            "speaking": [[0, 3], [21, 30], ...], # list of [start_token_id, end_token_id] for speaking mode
            ...
        }
        make sure all the token ranges are within len(hidden[layer])

        You take each hidden as input and predict its mode, and calculate the loss with the label.
        The label is 0 for listening mode and 1 for speaking mode.
        make sure(assert) the hidden size is of the moshi's vocab size

        the output path should be output_dir/hidden_mode_classifier_layer_{layer}.pt
        """
        

    def load(self, model_path):
        """Load the trained model from model_path"""
        pass

    def predict(self, input_hidden_path, output_path):
        """Predict the mode for a single input_hidden.pt and save the output to output_path
        The output should be a text json file with the format:
        {
            1: {
                "mode": "listening" or "speaking",
                "confidence": float
            }
            2 ...
        }
        """
        pass

def main():
    """The option should be stupid and simple
    --gen-dataset-hidden <dataset path>: generate the hidden for the mode class dataset
    --gen-sentence-hidden <sentence>: generate the hidden for a single sentence with --output <output path>
    --train-mode-classifier <dataset path>: train the mode classifier with the mode class dataset with --output <trained model path>
    --predict-mode <sentence> --model <trained model path> --output <output path>: predict the mode for a single sentence and save the output to output path
    only run static check as I don't have gpu
    """

"""

You are an expert PyTorch developer. Please write a complete, executable Python script to train a token-level linear classifier that predicts whether a language model is in "listening" (0) or "speaking" (1) mode based on its hidden states.

### 1. Dataset Structure & Loading
The dataset is located in a base directory (e.g., `Full-Duplex-Bench/data/mode_class/`). Inside this directory, there are multiple subdirectories, each containing:
- `input.json`: Contains metadata and token indices.
- `complete_sentence_hidden.pt`: A PyTorch tensor of hidden states.
- `incomplete_sentence_hidden.pt`: A PyTorch tensor of hidden states.

**Error Handling:** The script must verify the existence of the `.json` and both `.pt` files for every subdirectory. If any file is missing, log a clear error message and exit (quit) immediately.

### 2. Data Parsing & Labeling Logic
For each subdirectory, read `input.json`. Ignore the sentence text. Focus on the token indices mapping:
- `"listening": [[start_id, end_id], ...]` -> Assign label `0` (Listening)
- `"speaking": [[start_id, end_id], ...]` -> Assign label `1` (Speaking)

**CRITICAL - Inclusive Indexing:** The `[start_id, end_id]` intervals are INCLUSIVE. For example, `[3, 5]` means tokens at indices 3, 4, and 5 are in that mode. You must handle this correctly in Python (e.g., using `range(start, end + 1)`).

### 3. Model & Feature Specifications
- **Input:** A specific layer's hidden representation `hidden[layer]` loaded from the `.pt` files.
- **Model:** A simple linear layer (`nn.Linear`). 
- **Assertions:** 1. Ensure that all token indices extracted from the JSON are strictly `< len(hidden[layer])` to prevent index out-of-bounds errors.
  2. Assert that the feature dimension of the hidden state matches the expected Moshi vocabulary size. *(Note: use a configurable variable for this dimension).*

### 4. Training Loop
- Iterate through the valid hidden states token by token, extract the specific `layer` features, pass them through the linear classifier, and calculate the loss against the assigned 0/1 labels.
- Use an appropriate loss function for binary classification (e.g., `BCEWithLogitsLoss`).
- Include a basic optimizer (e.g., Adam) and a configurable number of epochs.

### 5. Output
Once training is complete, save the trained linear model's `state_dict` to:
`{output_dir}/hidden_mode_classifier_layer_{layer}.pt`

Please provide the fully commented, clean Python code.

"""