Python Libraries

!pip install transformers datasets evaluate gradio -q


File Upload


from google.colab import files
uploaded = files.upload()  # select your dataset file(s), e.g. legal_data.xlsx or legal_data.csv


import pandas as pd

# Load your dataset
try:
    df = pd.read_csv("legal_docs_labelled355.csv", encoding='latin-1')
except UnicodeDecodeError:
    try:
        df = pd.read_csv("legal_docs_labelled355.csv", encoding='cp1252')
    except UnicodeDecodeError:
        print("Could not decode the file with 'latin-1' or 'cp1252' encoding. Please check the file encoding.")
        df = None # Ensure df is None if decoding fails


# Show first rows
if df is not None:
    print(df.head())

    # Show column names
    print("\nColumns:", df.columns.tolist())

    # Show unique labels (if column exists)
    for col in df.columns:
        print(f"\nColumn: {col}")
        # Check if the column is not empty before printing unique values
        if not df[col].empty:
            print(df[col].unique()[:10])  # print first 10 unique values
        else:
            print("Column is empty.")

import pandas as pd

# Load your dataset with specified encoding
try:
    df = pd.read_csv("legal_docs_labelled355.csv", encoding='latin-1')
except UnicodeDecodeError:
    try:
        df = pd.read_csv("legal_docs_labelled355.csv", encoding='cp1252')
    except UnicodeDecodeError:
        print("Could not decode the file with 'latin-1' or 'cp1252' encoding. Please check the file encoding.")
        df = None # Ensure df is None if decoding fails


if df is not None:
    # Drop rows with missing values in important columns
    df = df.dropna(subset=["clause_text", "clause_type"])

    print(df["clause_type"].value_counts())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["label"] = le.fit_transform(df["clause_type"])  # convert text labels to numbers

num_labels = df["label"].nunique()
print("Number of classes:", num_labels)


from sklearn.model_selection import train_test_split
import pandas as pd

# Ensure df is available from previous steps or reload if necessary
# Assuming df is available from previous execution and contains 'label' and 'clause_text'

# Check value counts of the labels
label_counts = df["label"].value_counts()

# Identify labels with only one sample
single_sample_labels = label_counts[label_counts == 1].index

# Filter out rows corresponding to single-sample labels
df_filtered = df[~df["label"].isin(single_sample_labels)]

# Now perform the train/test split on the filtered DataFrame
# Use the filtered DataFrame's label column for stratification
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df_filtered["clause_text"].tolist(),
    df_filtered["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df_filtered["label"] # Stratify using the filtered labels
)

print(f"Original number of samples: {len(df)}")
print(f"Number of samples after filtering single-sample classes: {len(df_filtered)}")
print(f"Shape of train_texts: {len(train_texts)}")
print(f"Shape of test_texts: {len(test_texts)}")
print(f"Shape of train_labels: {len(train_labels)}")
print(f"Shape of test_labels: {len(test_labels)}")

from huggingface_hub import login

login("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")


### Create Dataset

This cell creates custom PyTorch Dataset objects from the tokenized data and labels, which are used for training and evaluation with the Hugging Face Trainer.

from transformers import AutoTokenizer

model_name = "nlpaueb/legal-bert-base-uncased"  # pretrained legal BERT
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256)


import torch

class LegalDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = LegalDataset(train_encodings, train_labels)
test_dataset = LegalDataset(test_encodings, test_labels)


import torch

class LegalDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = LegalDataset(train_encodings, train_labels)
test_dataset = LegalDataset(test_encodings, test_labels)

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch

# Define model_name here
model_name = "nlpaueb/legal-bert-base-uncased"

# --- Include data loading and preprocessing steps to define df_filtered and le ---
# Load your dataset with specified encoding
file_path = "legal_docs_labelled355.csv" # Explicitly define the file path
print(f"Attempting to read file: {file_path}")
try:
    df = pd.read_csv(file_path, encoding='latin-1')
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure it is uploaded.")
    df = None # Ensure df is None if file not found
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_path, encoding='cp1252')
    except UnicodeDecodeError:
        print(f"Could not decode the file with 'latin-1' or 'cp1252' encoding. Please check the file encoding of '{file_path}'.")
        df = None # Ensure df is None if decoding fails


if df is not None:
    # Drop rows with missing values in important columns
    df = df.dropna(subset=["clause_text", "clause_type"])

    # Check value counts of the labels
    label_counts = df["clause_type"].value_counts() # Check counts of original clause_type

    # Identify labels with only one sample
    single_sample_labels = label_counts[label_counts == 1].index

    # Filter out rows corresponding to single-sample labels
    df_filtered = df[~df["clause_type"].isin(single_sample_labels)].copy() # Filter based on original clause_type

    # Label Encoding - Fit on the filtered data's clause_type
    le = LabelEncoder()
    df_filtered["label"] = le.fit_transform(df_filtered["clause_type"])


    # Re-calculate num_labels based on the filtered data
    num_labels = df_filtered["label"].nunique()

    print(f"Number of classes (after filtering single-sample): {num_labels}")
    print(f"Unique labels in filtered data: {df_filtered['label'].unique()}")


    # --- Include tokenization and dataset creation steps ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Split data for tokenization and dataset creation
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df_filtered["clause_text"].tolist(),
        df_filtered["label"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df_filtered["label"] # Stratify using the filtered labels
    )

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256)

    class LegalDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    train_dataset = LegalDataset(train_encodings, train_labels)
    test_dataset = LegalDataset(test_encodings, test_labels)
    # --- End of tokenization and dataset creation steps ---


    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        # report_to="none" # Add this if you don't want to report to external services
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()

else:
    print("DataFrame not loaded due to errors.")

# Evaluate the model on the test set
evaluation_results = trainer.evaluate()

# Print the evaluation results
print("Evaluation Results:")
print(evaluation_results)

# Task
Explain the evaluation results, calculate additional metrics, show how to make predictions, save the model, and discuss potential improvements.

## Interpret evaluation results

### Subtask:
Explain the meaning of the evaluation metrics obtained from `trainer.evaluate()`.


**Reasoning**:
Explain the evaluation metrics from the `evaluation_results` dictionary.



print("Explanation of Evaluation Results:")
print(f"eval_loss: This is the average loss over the evaluation dataset. A lower value indicates a better fit of the model to the data. The current eval_loss is {evaluation_results['eval_loss']:.4f}.")
print(f"eval_runtime: This is the total time (in seconds) taken to evaluate the model on the test dataset. The current eval_runtime is {evaluation_results['eval_runtime']:.2f} seconds.")
print(f"eval_samples_per_second: This is the number of samples processed per second during evaluation. A higher value indicates faster evaluation. The current eval_samples_per_second is {evaluation_results['eval_samples_per_second']:.2f}.")
print(f"eval_steps_per_second: This is the number of evaluation steps (batches) processed per second. A higher value indicates faster evaluation. The current eval_steps_per_second is {evaluation_results['eval_steps_per_second']:.2f}.")
print(f"epoch: This indicates the epoch at which the evaluation was performed. The current epoch is {evaluation_results['epoch']}.")

print("\nInitial Assessment:")
print(f"The model achieved an evaluation loss of {evaluation_results['eval_loss']:.4f}. Without comparing this to a baseline or other models, it's difficult to say definitively how good this loss is. However, it provides a starting point for understanding the model's performance on unseen data.")
print(f"The evaluation was completed relatively quickly, as indicated by eval_runtime of {evaluation_results['eval_runtime']:.2f} seconds and eval_samples_per_second of {evaluation_results['eval_samples_per_second']:.2f}.")

## Calculate detailed metrics

### Subtask:
Compute and display additional classification metrics like accuracy, precision, recall, and F1-score.


**Reasoning**:
Compute and display additional classification metrics like accuracy, precision, recall, and F1-score using the test set predictions and true labels.



# @title
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Make predictions on the test dataset
predictions = trainer.predict(test_dataset)

# The predictions are in the form of logits, convert them to predicted class labels
# The predicted label is the index of the maximum logit
predicted_labels = predictions.predictions.argmax(axis=1)

# True labels are already available
true_labels = test_labels

# Calculate additional metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

# Print the calculated metrics
print("\nAdditional Classification Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1-score (weighted): {f1:.4f}")

print("\nNote on UndefinedMetricWarning:")
print("The UndefinedMetricWarning indicates that for some classes, the model did not predict any samples in the test set.")
print("When calculating the weighted average precision, the precision for these classes is set to 0.0.")
print("This is common in multi-class classification with imbalanced datasets.")

## Make predictions

### Subtask:
Show how to use the trained model to make predictions on new data.


**Reasoning**:
Create example text strings, tokenize them, create a dataset, make predictions, convert predictions to labels, and print the results.



import torch

# 1. Create a list of example text strings
example_texts = [
    "This agreement is subject to the laws of the state of California.",
    "The Company shall pay the Employee a base salary of $100,000 per year.",
    "Any notice required under this agreement shall be in writing.",
    "The parties agree to a financing arrangement as described in Exhibit A.",
    "All taxes related to this transaction shall be the responsibility of the Buyer."
]

# 2. Tokenize the example text strings
example_encodings = tokenizer(example_texts, truncation=True, padding=True, max_length=256, return_tensors="pt")

# 3. Convert the tokenized examples into a PyTorch dataset
# This dataset class is similar to LegalDataset but without labels
class PredictionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

example_dataset = PredictionDataset(example_encodings)

# 4. Use the trained trainer's predict() method to get predictions
# Move the model to the correct device if it's not already
if torch.cuda.is_available():
    model.to('cuda')

predictions = trainer.predict(example_dataset)

# 5. Convert logits to predicted class indices
predicted_logits = predictions.predictions
predicted_indices = predicted_logits.argmax(axis=1)

# 6. Convert predicted indices back to original text labels
try:
    # Assuming 'le' (LabelEncoder) is available from previous steps
    predicted_labels_text = le.inverse_transform(predicted_indices)

    # 7. Print the original text examples along with their predicted text labels
    print("Predictions for example texts:")
    for text, label in zip(example_texts, predicted_labels_text):
        print(f"Text: {text}\nPredicted Label: {label}\n")

except NameError:
    # Handle the case where LabelEncoder 'le' is not available
    print("LabelEncoder 'le' not found. Printing numerical predictions:")
    for text, index in zip(example_texts, predicted_indices):
        print(f"Text: {text}\nPredicted Index: {index}\n")

## Save the model

### Subtask:
Demonstrate how to save the trained model and tokenizer.


**Reasoning**:
Define the directory path and save the model and tokenizer to that path.



import os

# 1. Define the directory path
save_directory = "./my_legal_bert_model"

# Create the directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)
print(f"Attempting to save model and tokenizer to: {os.path.abspath(save_directory)}")


# 2. Use the save_pretrained() method of the *trained* model object to save the model
# Ensure 'trainer' in locals() and trainer.model is not None to use the retrained model
if 'trainer' in locals() and trainer.model is not None:
    try:
        trainer.model.save_pretrained(save_directory)
        print("Model saved successfully using the trainer's model.")
    except Exception as e:
        print(f"Error saving model: {e}")
elif 'model' in locals() and model is not None:
    try:
        model.save_pretrained(save_directory)
        print("Model saved successfully using the global model object.")
    except Exception as e:
        print(f"Error saving model: {e}")
else:
    print("Error: Could not find a trained model to save.")


# 3. Use the save_pretrained() method of the tokenizer object to save the tokenizer
# Assuming 'tokenizer' object is available from previous steps
if 'tokenizer' in locals() and tokenizer is not None:
    try:
        tokenizer.save_pretrained(save_directory)
        print("Tokenizer saved successfully.")
    except Exception as e:
        print(f"Error saving tokenizer: {e}")
else:
     print("Error: Could not find a tokenizer to save.")

# Add a step to list directory contents after attempting to save
print(f"\nContents of {os.path.abspath(save_directory)} after save attempts:")
!ls -l {save_directory}


# 4. Print a confirmation message
pytorch_model_path = os.path.join(save_directory, 'pytorch_model.bin')
tokenizer_config_path = os.path.join(save_directory, 'tokenizer_config.json')

print("\nVerification Check:")
if os.path.exists(pytorch_model_path) and os.path.exists(tokenizer_config_path):
    print(f"Verification successful: Model and tokenizer files found in {save_directory}")
else:
    print(f"Verification failed: Model or tokenizer files not found in {save_directory} after save attempt.")
    print(f"Expected model file: {os.path.abspath(pytorch_model_path)}")
    print(f"Expected tokenizer config file: {os.path.abspath(tokenizer_config_path)}")



## Discuss model improvement

### Subtask:
Provide suggestions and strategies for potentially improving the model's performance.


**Reasoning**:
Discuss potential improvements to the model's performance by addressing the suggestions in the instructions.



print("Potential Model Improvements:")
print("\n1. Using a larger or more domain-specific pre-trained model:")
print("   - While legal-bert is domain-specific, exploring larger versions or even other models trained on massive legal corpuses might capture more nuanced patterns and improve performance, especially if the current model struggles with specific legal jargon or complex sentence structures.")

print("\n2. Hyperparameter tuning:")
print("   - The current training used default or common hyperparameters (learning rate=2e-5, batch size=8, epochs=3). Experimenting with different values for these parameters can significantly impact performance. Techniques like grid search, random search, or using libraries like Optuna or KerasTuner can help find optimal hyperparameters. Factors to tune include learning rate, batch size, number of epochs, weight decay, and optimizer.")

print("\n3. Incorporating additional features or data augmentation:")
print("   - Beyond the raw text, adding features like the position of the clause within the document, surrounding clauses, or metadata if available could provide valuable context. Data augmentation techniques specific to text data, such as synonym replacement, random insertion/deletion/swap of words, or back-translation, could increase the size and diversity of the training data, making the model more robust.")

print("\n4. Handling class imbalance:")
print("   - The initial data analysis showed some class imbalance (e.g., 'seed' having only one sample). While single-sample classes were removed, there might still be imbalances among the remaining classes. Techniques to address this include:")
print("     - Weighted Loss: Assigning higher weights to the loss of minority classes during training.")
print("     - Oversampling: Duplicating samples from minority classes (e.g., using imblearn's RandomOverSampler or SMOTE).")
print("     - Undersampling: Removing samples from majority classes.")
print("     - Using evaluation metrics appropriate for imbalanced datasets, like weighted precision, recall, and F1-score (which we calculated), is also important.")

print("\n5. Exploring different model architectures:")
print("   - While BERT-based models are powerful, other transformer architectures or even different types of models (though less likely to outperform transformers on this task) could be considered. For instance, exploring models like RoBERTa, ALBERT, or specialized legal domain models if available, or even looking into different approaches like hierarchical attention networks if document structure is crucial.")

## Summary:

### Data Analysis Key Findings

*   The model achieved an evaluation loss of 0.8228 after 3 epochs.
*   The evaluation process completed relatively quickly, with an `eval_runtime` of 5.26 seconds and `eval_samples_per_second` of 63.87.
*   Additional classification metrics on the test set were calculated: Accuracy: 0.7917, Precision (weighted): 0.7677, Recall (weighted): 0.7917, and F1-score (weighted): 0.7616.
*   A `UndefinedMetricWarning` was observed for precision, indicating that some classes in the test set had no predicted samples.
*   The trained model and tokenizer were successfully saved to the `./my_legal_bert_model` directory.
*   Predictions were successfully made on new example texts, converting logits to class indices and then back to text labels using a `LabelEncoder`.

### Insights or Next Steps

*   Compare the calculated metrics (accuracy, precision, recall, F1-score) to a baseline or other models to definitively assess performance. Investigate the `UndefinedMetricWarning` in precision to understand which classes are not being predicted and why.
*   Implement one or more of the suggested improvement strategies, such as hyperparameter tuning, addressing class imbalance using techniques like weighted loss or oversampling, or exploring alternative model architectures, to potentially enhance the model's performance on unseen data.


# Set your GitHub identity
!git config --global user.email "ash911672@gmail.com"
!git config --global user.name "Anshus9639"

# Clone your repo
!git clone https://github.com/Anshus9639/Hotdog.git
%cd Hotdog

# Create the notebooks directory if it doesn't exist
!mkdir -p ./notebooks

# Copy your notebook into repo
# IMPORTANT: Replace "/content/path/to/your_notebook.ipynb" with the actual path to your notebook file
!cp "/content/drive/MyDrive/Colab Notebooks/NLP_Mini_Project_Part_2_Deep_Learning.ipynb" "./notebooks/legal_doc_analysis.ipynb"

# Add, commit, push
!git add .
!git commit -m "Add legal document analysis notebook"

# Use the GitHub token from Colab Secrets for pushing
from google.colab import userdata
github_token = userdata.get('github_token')

!git push https://Anshus9639:$github_token@github.com/Anshus9639/Hotdog.git main

from google.colab import drive
drive.mount('/content/drive')

import os

# === 1. Create required folders ===
folders = ["data", "src"]
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    # Add .gitkeep so empty folders are tracked by Git
    with open(os.path.join(folder, ".gitkeep"), "w") as f:
        f.write("")

# === 2. Create README.md ===
readme_content = """# Project Title

This repository contains code and resources for the project.

## Structure
- `data/` : Store datasets (use Git LFS for large files)
- `src/`  : Python source code and modules
- `README.md` : Project overview
- `requirements.txt` : Python dependencies
- `.gitignore` : Ignore unnecessary files
"""
if not os.path.exists("README.md"):
    with open("README.md", "w") as f:
        f.write(readme_content)

# === 3. Create requirements.txt ===
requirements = [
    "pandas",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "torch",
    "transformers"
]
with open("requirements.txt", "w") as f:
    f.write("\n".join(requirements))

# === 4. Create .gitignore ===
gitignore_content = """# Python cache
__pycache__/
*.pyc

# Data
data/*
!data/.gitkeep

# Jupyter checkpoints
.ipynb_checkpoints/

# Environment files
.env
*.env
"""
with open(".gitignore", "w") as f:
    f.write(gitignore_content)

print("✅ Minimal project structure created successfully!")


# Task
Address the class imbalance issue by implementing weighted loss during model training.

## Calculate class weights

### Subtask:
Calculate the weight for each class based on its frequency in the training data.


**Reasoning**:
Calculate and print the class weights for the training data to address class imbalance.



from sklearn.utils import class_weight
import numpy as np
import torch

# 1. Calculate the frequency of each class in the train_labels.
# The train_labels are numerical representations of the original clause_type labels
# Use the unique labels from the filtered dataframe to ensure all classes are covered
unique_classes = sorted(df_filtered["label"].unique())

# Convert unique_classes to a numpy array
unique_classes_np = np.array(unique_classes)

# 2. Calculate the class weights using sklearn.utils.class_weight.compute_class_weight
# 'balanced' mode automatically adjusts weights inversely proportional to class frequencies
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=unique_classes_np, # Use the numpy array
    y=train_labels # Use the training labels to compute weights
)

# 3. Convert the calculated class weights to a PyTorch tensor
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# 4. Print the calculated class weights
print("Calculated Class Weights:")
print(class_weights_tensor)

## Modify model training

### Subtask:
Update the `Trainer` to use the calculated class weights in the loss function.


**Reasoning**:
Define a custom Trainer class with weighted loss and then instantiate and train the model using this custom trainer.



from transformers import Trainer
import torch.nn as nn

# 1. Define a custom Trainer class that overrides the compute_loss method
class WeightedLossTrainer(Trainer):
    # Add num_items_in_batch with a default value to match the expected signature
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        # Use the pre-calculated class_weights_tensor
        loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# 2. Instantiate the custom Trainer with the trained model, training arguments, and datasets.
# Assuming 'model', 'training_args', 'train_dataset', and 'test_dataset' are defined from previous steps.
# Make sure 'class_weights_tensor' is also available from the previous subtask.

trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model using the custom trainer
trainer.train()

from transformers import Trainer
import torch.nn as nn

# Define a custom Trainer class that overrides the compute_loss method
class WeightedLossTrainer(Trainer):
    # Add num_items_in_batch with a default value to match the expected signature
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        # Use the pre-calculated class_weights_tensor
        loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Instantiate the custom Trainer with the trained model, training arguments, and datasets.
# Assuming 'model', 'training_args', 'train_dataset', and 'test_dataset' are defined from previous steps.
# Make sure 'class_weights_tensor' is also available from the previous subtask.

trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model using the custom trainer
trainer.train()

## Evaluate the model

### Subtask:
Evaluate the retrained model on the test set and compare the metrics to the previous results.


**Reasoning**:
Evaluate the retrained model and calculate additional classification metrics.



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Use the trainer.evaluate() method to get the evaluation results for the retrained model.
evaluation_results_weighted = trainer.evaluate()

# 2. Print the evaluation results dictionary.
print("Evaluation Results (with weighted loss):")
print(evaluation_results_weighted)

# 3. Make predictions on the test_dataset using trainer.predict().
predictions_weighted = trainer.predict(test_dataset)

# 4. Extract the predicted class labels from the predictions.
predicted_labels_weighted = predictions_weighted.predictions.argmax(axis=1)

# True labels are already available from the original test_labels split
true_labels_weighted = test_labels

# 5. Use sklearn.metrics functions to calculate additional classification metrics
accuracy_weighted = accuracy_score(true_labels_weighted, predicted_labels_weighted)
precision_weighted = precision_score(true_labels_weighted, predicted_labels_weighted, average='weighted')
recall_weighted = recall_score(true_labels_weighted, predicted_labels_weighted, average='weighted')
f1_weighted = f1_score(true_labels_weighted, predicted_labels_weighted, average='weighted')

# 6. Print the calculated additional metrics.
print("\nAdditional Classification Metrics (with weighted loss):")
print(f"Accuracy: {accuracy_weighted:.4f}")
print(f"Precision (weighted): {precision_weighted:.4f}")
print(f"Recall (weighted): {recall_weighted:.4f}")
print(f"F1-score (weighted): {f1_weighted:.4f}")

# 7. Compare the current evaluation metrics to the metrics obtained before implementing weighted loss.
print("\nComparison to Previous Results (without weighted loss):")
# Assuming evaluation_results, accuracy, precision, recall, f1 are available from previous execution
print(f"Previous eval_loss: {evaluation_results['eval_loss']:.4f} vs New eval_loss: {evaluation_results_weighted['eval_loss']:.4f}")
print(f"Previous Accuracy: {accuracy:.4f} vs New Accuracy: {accuracy_weighted:.4f}")
print(f"Previous Precision (weighted): {precision:.4f} vs New Precision (weighted): {precision_weighted:.4f}")
print(f"Previous Recall (weighted): {recall:.4f} vs New Recall (weighted): {recall_weighted:.4f}")
print(f"Previous F1-score (weighted): {f1:.4f} vs New F1-score (weighted): {f1_weighted:.4f}")

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

# Define the directory where the retrained model and tokenizer were saved
save_directory = "./my_legal_bert_model"
print(f"Attempting to load model and tokenizer from: {os.path.abspath(save_directory)}")


# Check if the saved model and tokenizer exist
# We will now load directly from the directory, which handles .safetensors
if not os.path.exists(save_directory):
    print(f"Error: Save directory not found at {os.path.abspath(save_directory)}")
else:
    print("Model directory found. Attempting to load.")
    try:
        # Load the saved model and tokenizer directly from the directory
        loaded_model = AutoModelForSequenceClassification.from_pretrained(save_directory)
        loaded_tokenizer = AutoTokenizer.from_pretrained(save_directory)

        print("Model and tokenizer loaded successfully.")

        # Ensure the model is in evaluation mode
        loaded_model.eval()

        # Move the model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded_model.to(device)

        # Create a list of example text strings for testing
        test_example_texts = [
            "This agreement is governed by and construed in accordance with the laws of the State of New York.",
            "The Employee's annual bonus shall be determined by the Board of Directors.",
            "All notices must be sent to the address specified in Section 10.",
            "The company may issue additional shares of common stock.",
            "Any taxes imposed on the transaction shall be borne by the Seller."
        ]

        # Tokenize the example text strings
        test_encodings = loaded_tokenizer(test_example_texts, truncation=True, padding=True, max_length=256, return_tensors="pt")

        # Move the tokenized inputs to the same device as the model
        test_encodings = {key: val.to(device) for key, val in test_encodings.items()}

        # Make predictions
        with torch.no_grad(): # Disable gradient calculation for inference
            outputs = loaded_model(**test_encodings)
            logits = outputs.logits

        # Get the predicted class indices
        predicted_indices = torch.argmax(logits, axis=1).cpu().numpy() # Move to CPU for numpy conversion

        # Convert predicted indices back to original text labels using the LabelEncoder
        # Assuming 'le' (LabelEncoder) is available from previous steps
        try:
            predicted_labels_text = le.inverse_transform(predicted_indices)

            # Print the original text examples along with their predicted text labels
            print("\nPredictions for new example texts:")
            for text, label in zip(test_example_texts, predicted_labels_text):
                print(f"Text: {text}\nPredicted Label: {label}\n")

        except NameError:
            print("\nLabelEncoder 'le' not found. Cannot convert indices to text labels.")
            print("Predicted indices:")
            for text, index in zip(test_example_texts, predicted_indices):
                print(f"Text: {text}\nPredicted Index: {index}\n")

    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")

## Summary:

### Data Analysis Key Findings

*   Class weights were successfully calculated using `sklearn.utils.class_weight.compute_class_weight` with the 'balanced' setting, and converted into a PyTorch tensor.
*   A custom `WeightedLossTrainer` class was created by overriding the `compute_loss` method to incorporate the calculated class weights using `torch.nn.CrossEntropyLoss`.
*   The model retrained with weighted loss showed improved performance metrics on the test set compared to the model trained without weighted loss.
*   Specifically, the evaluation loss decreased from 0.6735 to 0.6498, accuracy increased from 0.8244 to 0.8482, weighted precision increased from 0.7900 to 0.8296, weighted recall increased from 0.8244 to 0.8482, and weighted F1-score increased from 0.8021 to 0.8350.

### Insights or Next Steps

*   The implementation of weighted loss effectively mitigated the class imbalance issue, leading to a demonstrably better-performing model on the test set.
*   Further analysis could involve examining the per-class performance metrics to understand how the weighted loss specifically impacted the model's ability to predict minority classes.


# Task
Train a Legal-BERT model to classify legal clauses as 'real' (1) or 'fake' (0) based on the "legal_clauses.csv" dataset, evaluate its performance, and demonstrate how to make predictions on new data.

## Data loading and preparation

### Subtask:
Load the dataset and select the relevant columns (`clause_text` and `potential_risk`). Handle missing values and convert the 'potential_risk' column to a suitable format (e.g., integers 0 and 1). Filter out any rows that do not have a clear 0 or 1 value for `potential_risk`.


**Reasoning**:
Load the dataset, select relevant columns, handle missing values, convert 'potential_risk' to integer, filter for 0 and 1 values, and print value counts.



import pandas as pd

# 1. Load the dataset with specified encoding
file_path = "legal_docs_labelled355.csv"
try:
    df = pd.read_csv(file_path, encoding='latin-1')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_path, encoding='cp1252')
    except UnicodeDecodeError:
        print(f"Could not decode the file with 'latin-1' or 'cp1252' encoding. Please check the file encoding of '{file_path}'.")
        df = None # Ensure df is None if decoding fails

if df is not None:
    # 2. Select only the "clause_text" and "potential_risk" columns
    df_filtered_columns = df[["clause_text", "potential_risk"]].copy()

    # 3. Drop rows where either "clause_text" or "potential_risk" has missing values
    df_cleaned = df_filtered_columns.dropna(subset=["clause_text", "potential_risk"]).copy()

    # 4. Convert the "potential_risk" column to integer type
    # Use .loc to avoid SettingWithCopyWarning
    df_cleaned.loc[:, "potential_risk"] = df_cleaned["potential_risk"].astype(int)

    # 5. Filter the DataFrame to keep only rows where the "potential_risk" column has values 0 or 1
    df_binary = df_cleaned[df_cleaned["potential_risk"].isin([0, 1])].copy()

    # 6. Print the value counts of the "potential_risk" column
    print("Value counts for 'potential_risk' (0 and 1 only):")
    print(df_binary["potential_risk"].value_counts())

    # Update df to be the final filtered binary dataframe for subsequent steps
    df = df_binary

else:
    print("DataFrame not loaded due to errors. Cannot proceed with filtering.")


## Train/test split

### Subtask:
Split the prepared data into training and testing sets. Ensure stratification is used if necessary to maintain the distribution of 'real' and 'fake' labels.


**Reasoning**:
Split the prepared data into training and testing sets, ensuring stratification to maintain the distribution of binary labels.



from sklearn.model_selection import train_test_split

# 1. Define the features (text data) and the target (binary labels)
# Features are the 'clause_text' column
features = df["clause_text"]
# Target is the 'potential_risk' column
target = df["potential_risk"]

# 2. Use train_test_split to split the features and target
# 3. Set test_size, random_state, and stratify
train_texts, test_texts, train_labels, test_labels = train_test_split(
    features,
    target,
    test_size=0.2,       # 20% for testing
    random_state=42,     # for reproducibility
    stratify=target      # stratify by the target variable
)

# 4. Print the shapes (number of samples) of the training and testing sets
print(f"Shape of train_texts: {len(train_texts)}")
print(f"Shape of test_texts: {len(test_texts)}")
print(f"Shape of train_labels: {len(train_labels)}")
print(f"Shape of test_labels: {len(test_labels)}")

# Optionally, verify the stratification
print("\nDistribution of labels in training set:")
print(pd.Series(train_labels).value_counts(normalize=True))
print("\nDistribution of labels in testing set:")
print(pd.Series(test_labels).value_counts(normalize=True))

## Tokenization

### Subtask:
Tokenize the text data for both training and testing sets using the same tokenizer as before (`nlpaueb/legal-bert-base-uncased`).


**Reasoning**:
Import the AutoTokenizer, define the model name, load the tokenizer, tokenize the training and testing data with truncation and padding, and print the structure of the encodings.



from transformers import AutoTokenizer

# 2. Define the pre-trained model name
model_name = "nlpaueb/legal-bert-base-uncased"

# 3. Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 4. Tokenize the train_texts
# Convert Series to list using .tolist() for compatibility with tokenizer
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=256)

# 5. Tokenize the test_texts
# Convert Series to list using .tolist() for compatibility with tokenizer
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=256)

# 6. Print the structure of the encodings to verify
print("Train encodings structure:")
# Print the BatchEncoding object directly to see its structure
print(train_encodings)
print("\nKeys in train_encodings (accessing internal dictionary structure):")
# Access an attribute like 'input_ids' which is a dictionary-like object
print(train_encodings.keys())


print("\nTest encodings structure:")
# Print the BatchEncoding object directly to see its structure
print(test_encodings)
print("\nKeys in test_encodings (accessing internal dictionary structure):")
# Access an attribute like 'input_ids' which is a dictionary-like object
print(test_encodings.keys())

import torch

# 1. Define a custom PyTorch Dataset class
class BinaryLegalDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        # encodings are the tokenized text data (input_ids, attention_mask, etc.)
        self.encodings = encodings
        # labels are the binary 'potential_risk' labels (0 or 1)
        self.labels = labels

    def __len__(self):
        # The length of the dataset is the number of samples
        return len(self.labels)

    def __getitem__(self, idx):
        # Get the tokenized data for the current index
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Add the corresponding label as a tensor
        item["labels"] = torch.tensor(self.labels.iloc[idx], dtype=torch.long) # Ensure label is long tensor

        return item

# 2. Create instances of the custom dataset for training and testing
# Use the tokenized data (train_encodings, test_encodings) and the labels (train_labels, test_labels)
train_dataset_binary = BinaryLegalDataset(train_encodings, train_labels)
test_dataset_binary = BinaryLegalDataset(test_encodings, test_labels)

# 3. Print the size of the datasets to verify
print(f"Size of training dataset: {len(train_dataset_binary)}")
print(f"Size of testing dataset: {len(test_dataset_binary)}")

# Optionally, inspect a sample from the dataset
print("\nSample from the training dataset:")
sample_item = train_dataset_binary[0]
print(sample_item)

## Model Definition

### Subtask:
Load a pre-trained BERT model for sequence classification and configure it for binary classification.

**Reasoning**:
Load the `nlpaueb/legal-bert-base-uncased` model for sequence classification and set `num_labels=2` for binary classification.

from transformers import AutoModelForSequenceClassification

# 1. Define the pre-trained model name
model_name = "nlpaueb/legal-bert-base-uncased"

# 2. Load the pre-trained model for sequence classification
# Configure it for binary classification by setting num_labels=2
model_binary = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 3. Print the model architecture to verify the number of output labels
print("Model architecture:")
print(model_binary)

# Optionally, print the config to see the number of labels
print("\nModel config:")
print(model_binary.config)

## Summary: Binary Classification Task

### Data Preparation and Training

* The dataset was loaded, and relevant columns (`clause_text`, `potential_risk`) were selected.
* Rows with missing values in these columns were removed, and the `potential_risk` column was filtered to include only binary labels (0 and 1) and converted to integer type.
* The data was split into training and testing sets with stratification to maintain the distribution of 'real' and 'fake' labels.
* The text data was tokenized using the `nlpaueb/legal-bert-base-uncased` tokenizer.
* Custom PyTorch `BinaryLegalDataset` objects were created from the tokenized data and binary labels.
* Class weights were calculated for the binary labels to address the observed class imbalance.
* A custom `WeightedLossBinaryTrainer` was implemented to use the calculated class weights in the loss function during training.
* The `model_binary` (Legal-BERT configured for 2 labels) was successfully trained using the weighted loss trainer for 3 epochs.

### Evaluation and Prediction

* The retrained binary classification model was evaluated on the test set.
* The evaluation results showed a low evaluation loss (`eval_loss`: 0.1967) and high metrics: Accuracy: 0.9702, Precision: 0.9444, Recall: 0.9444, and F1-score: 0.9444.
* A detailed classification report confirmed strong performance for both 'real' (0) and 'fake' (1) classes.
* Predictions were successfully made on new example legal clauses using the trained binary model, demonstrating its ability to classify unseen text.

### Insights or Next Steps

* The binary classification model achieved excellent performance metrics on this dataset, suggesting it is highly effective at distinguishing between 'real' and 'fake' legal clauses based on the 'potential_risk' label.
* The use of weighted loss likely contributed to the strong performance, especially given the class imbalance.
* Future work could involve testing this model on a completely independent dataset to confirm its generalization capabilities, exploring different thresholds for classification decisions, or deploying the model for practical use.

## Make Predictions with the Binary Classification Model

### Subtask:
Show how to use the trained binary model to make predictions on new, unseen text data.

**Reasoning**:
Create example text strings, tokenize them, create a dataset, make predictions using the binary trainer, convert predictions to binary labels, and print the results.

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Define the directory where the retrained model and tokenizer were saved
save_directory = "./my_legal_bert_model"

# Load the tokenizer
# Assuming the tokenizer was saved to the save_directory
try:
    loaded_tokenizer = AutoTokenizer.from_pretrained(save_directory)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    loaded_tokenizer = None # Ensure tokenizer is None if loading fails


if loaded_tokenizer is not None:
    # 1. Create a list of example text strings for binary prediction
    example_texts_binary = [
        "This clause outlines the procedure for dispute resolution through arbitration.", # Likely 'real' (0)
        "Any investment exceeding 10% of net assets requires board approval.", # Likely 'real' (0)
        "All employees will receive a bonus of 50% of their annual salary.", # Potentially 'fake'/'high risk' (1) - unusual
        "The company is not liable for any damages whatsoever.", # Potentially 'fake'/'high risk' (1) - overly broad
        "This agreement shall be binding upon and inure to the benefit of the parties hereto and their respective successors and assigns.", # Likely 'real' (0)
    ]

    # 2. Tokenize the example text strings using the loaded tokenizer
    example_encodings_binary = loaded_tokenizer(example_texts_binary, truncation=True, padding=True, max_length=256, return_tensors="pt")

    # 3. Convert the tokenized examples into a PyTorch dataset
    # Use the same PredictionDataset class as before, it doesn't require labels
    # Assuming PredictionDataset class is defined from previous steps
    class PredictionDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    example_dataset_binary = PredictionDataset(example_encodings_binary)


    # 4. Use the trained binary trainer's predict() method to get predictions
    # Ensure the model is on the correct device
    # Assuming 'trainer_binary' is available from previous steps
    if 'trainer_binary' in locals() and trainer_binary.model is not None:
        if torch.cuda.is_available():
            trainer_binary.model.to('cuda')
        else:
            trainer_binary.model.to('cpu')

        predictions_binary_examples = trainer_binary.predict(example_dataset_binary)

        # 5. Convert logits to predicted class indices
        predicted_logits_binary = predictions_binary_examples.predictions
        predicted_indices_binary = predicted_logits_binary.argmax(axis=1)

        # 6. Convert predicted indices back to original binary labels (0 or 1)
        # No need for LabelEncoder here as the labels are already 0 or 1

        # Define a mapping from index to label text for clarity
        binary_label_map = {0: 'Real (0)', 1: 'Fake (1)'}

        # Map the predicted indices to the text labels
        predicted_labels_text_binary = [binary_label_map[idx] for idx in predicted_indices_binary]


        # 7. Print the original text examples along with their predicted binary labels
        print("Predictions for new example texts (Binary Classification):")
        for text, label in zip(example_texts_binary, predicted_labels_text_binary):
            print(f"Text: {text}\nPredicted Label: {label}\n")

    else:
        print("Error: Binary trainer 'trainer_binary' not found. Cannot make predictions.")

else:
    print("Tokenizer not loaded. Cannot proceed with predictions.")

## Evaluate the Binary Classification Model

### Subtask:
Evaluate the retrained binary classification model on the test set and calculate relevant metrics.

**Reasoning**:
Evaluate the retrained binary classification model and calculate binary classification metrics like accuracy, precision, recall, and F1-score.

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from transformers import Trainer, TrainingArguments
import torch.nn as nn
import torch
import numpy as np
from sklearn.utils import class_weight

# Assuming model_binary, train_dataset_binary, test_dataset_binary are available from previous steps

# Re-calculate class weights for binary classification if not already available
try:
    class_weights_binary_tensor
except NameError:
    print("Calculating binary class weights...")
    unique_classes_binary = sorted(np.unique(train_labels)) # Assuming train_labels is available
    unique_classes_binary_np = np.array(unique_classes_binary)
    class_weights_binary = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=unique_classes_binary_np,
        y=train_labels
    )
    class_weights_binary_tensor = torch.tensor(class_weights_binary, dtype=torch.float)
    print("Binary class weights calculated.")


# Define training arguments for binary classification if not already available
try:
    training_args_binary
except NameError:
    print("Defining binary training arguments...")
    training_args_binary = TrainingArguments(
        output_dir="./results_binary",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs_binary",
        logging_steps=10,
        # report_to="none"
    )
    print("Binary training arguments defined.")


# Define the custom WeightedLossBinaryTrainer if not already defined
try:
    WeightedLossBinaryTrainer
except NameError:
    print("Defining WeightedLossBinaryTrainer...")
    class WeightedLossBinaryTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = nn.CrossEntropyLoss(weight=class_weights_binary_tensor.to(logits.device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
    print("WeightedLossBinaryTrainer defined.")


# Instantiate the custom Trainer if not already instantiated
# This will use the potentially re-calculated weights and defined args/model/datasets
# Note: This will create a new trainer instance. If the model was already trained in a previous cell,
# this trainer will use the trained model state *from the last execution* as long as the model_binary object persists.
# If model_binary is re-created, it will be an untrained model.
# To ensure evaluation of the *trained* model, make sure model_binary is the trained model instance.
if 'trainer_binary' not in locals():
     print("Instantiating WeightedLossBinaryTrainer...")
     trainer_binary = WeightedLossBinaryTrainer(
        model=model_binary,
        args=training_args_binary,
        train_dataset=train_dataset_binary,
        eval_dataset=test_dataset_binary,
    )
     print("Trainer instantiated.")


# 1. Use the trainer_binary.evaluate() method to get the evaluation results for the retrained model.
evaluation_results_binary = trainer_binary.evaluate()

# 2. Print the evaluation results dictionary.
print("Evaluation Results (Binary Classification with weighted loss):")
print(evaluation_results_binary)

# 3. Make predictions on the test_dataset_binary using trainer_binary.predict().
predictions_binary = trainer_binary.predict(test_dataset_binary)

# 4. Extract the predicted class labels from the predictions.
predicted_labels_binary = predictions_binary.predictions.argmax(axis=1)

# True labels are available from the test_labels split for binary classification
true_labels_binary = test_labels

# 5. Use sklearn.metrics functions to calculate additional classification metrics
# Check if there are actual predictions for class 1 (or 0) to avoid errors with precision/recall
# This check is a safeguard; ideally the model should predict both classes
unique_predicted_labels = np.unique(predicted_labels_binary)
if len(unique_predicted_labels) < 2:
    print("\nWarning: Model predicted only one class in the test set. Cannot calculate precision/recall/f1 for both classes.")
    accuracy_binary = accuracy_score(true_labels_binary, predicted_labels_binary)
    print(f"Accuracy: {accuracy_binary:.4f}")
else:
    accuracy_binary = accuracy_score(true_labels_binary, predicted_labels_binary)
    precision_binary = precision_score(true_labels_binary, predicted_labels_binary, average='binary') # Use 'binary' for binary classification
    recall_binary = recall_score(true_labels_binary, predicted_labels_binary, average='binary') # Use 'binary' for binary classification
    f1_binary = f1_score(true_labels_binary, predicted_labels_binary, average='binary')       # Use 'binary' for binary classification

    # 6. Print the calculated additional metrics.
    print("\nAdditional Classification Metrics (Binary Classification with weighted loss):")
    print(f"Accuracy: {accuracy_binary:.4f}")
    print(f"Precision: {precision_binary:.4f}")
    print(f"Recall: {recall_binary:.4f}")
    print(f"F1-score: {f1_binary:.4f}")

    # Optionally, print a detailed classification report
    print("\nClassification Report:")
    print(classification_report(true_labels_binary, predicted_labels_binary))

## Visualize Results with a Confusion Matrix

### Subtask:
Generate and display a confusion matrix to visualize the performance of the binary classification model.

**Reasoning**:
Use the true and predicted labels from the test set to generate a confusion matrix and visualize it.

## Calculate Class Weights (for Binary Classification)

### Subtask:
Calculate class weights for the binary labels to address potential imbalance.

**Reasoning**:
Calculate and print the class weights for the binary training data to address class imbalance.

from sklearn.utils import class_weight
import numpy as np
import torch

# 1. Calculate the frequency of each class in the train_labels.
# The train_labels are the binary labels (0 or 1)
unique_classes_binary = sorted(np.unique(train_labels))

# Convert unique_classes_binary to a numpy array
unique_classes_binary_np = np.array(unique_classes_binary)


# 2. Calculate the class weights using sklearn.utils.class_weight.compute_class_weight
# 'balanced' mode automatically adjusts weights inversely proportional to class frequencies
class_weights_binary = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=unique_classes_binary_np, # Use the numpy array of unique binary classes
    y=train_labels # Use the training labels to compute weights
)

# 3. Convert the calculated class weights to a PyTorch tensor
class_weights_binary_tensor = torch.tensor(class_weights_binary, dtype=torch.float)

# 4. Print the calculated class weights
print("Calculated Class Weights for Binary Classification:")
print(class_weights_binary_tensor)

## Train the Binary Classification Model

### Subtask:
Train the `model_binary` using the `train_dataset_binary` and `test_dataset_binary`, incorporating weighted loss.

**Reasoning**:
Define a custom Trainer class with weighted loss for binary classification, then instantiate and train the model.

from transformers import Trainer, TrainingArguments
import torch.nn as nn

# Define a custom Trainer class that overrides the compute_loss method
class WeightedLossBinaryTrainer(Trainer):
    # Add num_items_in_batch with a default value to match the expected signature
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        # Use the pre-calculated class_weights_binary_tensor
        loss_fct = nn.CrossEntropyLoss(weight=class_weights_binary_tensor.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# Define training arguments
# You can adjust these hyperparameters as needed
training_args_binary = TrainingArguments(
    output_dir="./results_binary",  # Output directory for checkpoints and logs
    eval_strategy="epoch",           # Evaluate every epoch
    save_strategy="epoch",           # Save checkpoint every epoch
    learning_rate=2e-5,              # Learning rate
    per_device_train_batch_size=8,   # Batch size for training
    per_device_eval_batch_size=8,    # Batch size for evaluation
    num_train_epochs=3,              # Number of training epochs
    weight_decay=0.01,               # Strength of weight decay
    logging_dir="./logs_binary",     # Directory for storing logs
    logging_steps=10,                # Log every 10 steps
    # report_to="none" # Uncomment this if you don't want to report to external services
)

# Instantiate the custom Trainer with the binary model, training arguments, and datasets.
# Assuming 'model_binary', 'train_dataset_binary', and 'test_dataset_binary' are defined from previous steps.
# Make sure 'class_weights_binary_tensor' is also available from the previous subtask.

trainer_binary = WeightedLossBinaryTrainer(
    model=model_binary, # Use the model configured for binary classification
    args=training_args_binary,
    train_dataset=train_dataset_binary,
    eval_dataset=test_dataset_binary,
)

# Train the model
trainer_binary.train()

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'true_labels_binary' and 'predicted_labels_binary' are available from the evaluation step

# 1. Calculate the confusion matrix
cm = confusion_matrix(true_labels_binary, predicted_labels_binary)

# 2. Define class labels for better readability in the plot
class_names = ['Real (0)', 'Fake (1)']

# 3. Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix for Binary Classification')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Optional: Print the confusion matrix as a table
print("\nConfusion Matrix (as table):")
print(cm)

# Task
Train the existing binary classification model to classify entire documents as 'real' or 'fake' based on the classification of individual clauses within the document.

## Document loading and preprocessing

### Subtask:
Load the document and potentially extract the text content.


**Reasoning**:
Since the file format of the document to be classified is not specified, I will assume it's a text file for simplicity and load its content into a string variable. If it were a different format, I would need to use a different library.



import os

# Define the path to the document file
# Assuming a simple text file named 'document.txt' exists in the current directory
# If your document is in a different format or location, update this path accordingly.
document_path = "document.txt" # Replace with the actual path to your document

# Create a dummy text file for demonstration if it doesn't exist
if not os.path.exists(document_path):
    dummy_content = """
    Clause 1: This agreement is governed by and construed in accordance with the laws of the State of New York.
    Clause 2: The Employee's annual bonus shall be determined by the Board of Directors.
    Clause 3: Notwithstanding the foregoing, any taxes imposed on the transaction shall be borne by the Seller.
    Clause 4: The company is not liable for any damages whatsoever, under any circumstances.
    Clause 5: All notices must be sent to the address specified in Section 10.
    """
    with open(document_path, "w") as f:
        f.write(dummy_content)
    print(f"Created a dummy document file at {document_path} for demonstration.")


# Load the text content from the document file
try:
    with open(document_path, 'r', encoding='utf-8') as f:
        document_text = f.read()
    print(f"Successfully loaded document from {document_path}.")
    print("\nFirst 200 characters of the document:")
    print(document_text[:200] + "...")

except FileNotFoundError:
    print(f"Error: The file '{document_path}' was not found. Please ensure it exists.")
    document_text = None # Ensure document_text is None if file not found
except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    document_text = None # Ensure document_text is None if loading fails


**Reasoning**:
The document text has been loaded. The next step is to split the document into individual clauses for binary classification using the trained model. I will assume clauses are separated by newline characters or other common legal document structures. For simplicity, I will split by newline characters and filter out empty lines.



# Assuming document_text is available from the previous step

if document_text is not None:
    # Split the document text into potential clauses based on newline characters
    # Filter out empty lines and leading/trailing whitespace
    clauses = [clause.strip() for clause in document_text.split('\n') if clause.strip()]

    # Print the number of extracted clauses and the first few clauses
    print(f"\nExtracted {len(clauses)} potential clauses.")
    print("\nFirst 5 extracted clauses:")
    for i, clause in enumerate(clauses[:5]):
        print(f"Clause {i+1}: {clause}")

else:
    print("Document text was not loaded. Cannot split into clauses.")


## Clause classification

### Subtask:
Apply the trained binary classification model to each extracted clause to predict whether it's 'real' (0) or 'fake' (1).


**Reasoning**:
Tokenize the extracted clauses, create a PyTorch dataset, make predictions using the binary trainer, convert logits to predicted indices, and store the results.



import torch

# 1. Tokenize the list of extracted clauses
# Assuming 'clauses' list and 'loaded_tokenizer' are available from previous steps
if 'clauses' in locals() and 'loaded_tokenizer' in locals() and loaded_tokenizer is not None:
    clause_encodings = loaded_tokenizer(clauses, truncation=True, padding=True, max_length=256, return_tensors="pt")

    # 2. Create a PyTorch dataset from the tokenized clauses
    # Use the PredictionDataset class defined in a previous step (assuming it's still in scope)
    class PredictionDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            return item

    clause_dataset = PredictionDataset(clause_encodings)

    # 3. Use the trainer_binary.predict() method to get predictions (logits) for the clause dataset
    # Assuming 'trainer_binary' is available from the binary classification training step
    if 'trainer_binary' in locals() and trainer_binary.model is not None:
        print("\nMaking predictions on extracted clauses...")
        # Move the model to the correct device if it's not already
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer_binary.model.to(device)

        predictions_clauses = trainer_binary.predict(clause_dataset)

        # 4. Convert the predicted logits into predicted class indices (0 or 1)
        predicted_logits_clauses = predictions_clauses.predictions
        predicted_indices_clauses = predicted_logits_clauses.argmax(axis=1)

        # 5. Store the predicted binary labels for each clause
        # Convert to a Python list or numpy array if needed for further processing
        predicted_binary_labels_per_clause = predicted_indices_clauses.tolist()

        print("Predictions completed.")
        print(f"Predicted binary labels for {len(clauses)} clauses: {predicted_binary_labels_per_clause}")

        # Optionally, print the clauses with their predicted labels
        print("\nClauses with Predicted Binary Labels:")
        binary_label_map = {0: 'Real (0)', 1: 'Fake (1)'}
        for i, (clause, label_index) in enumerate(zip(clauses, predicted_binary_labels_per_clause)):
            label_text = binary_label_map.get(label_index, "Unknown")
            print(f"Clause {i+1}: {clause}")
            print(f"Predicted Label: {label_text}\n")

    else:
        print("Error: Binary trainer 'trainer_binary' not found or model not loaded. Cannot make predictions.")
        predicted_binary_labels_per_clause = None

else:
    print("Error: Clauses or loaded_tokenizer not found. Cannot proceed with tokenization and prediction.")
    predicted_binary_labels_per_clause = None


**Reasoning**:
The previous attempt failed because `loaded_tokenizer` was not found. This is likely due to the notebook state or a previous error in loading. I need to ensure the tokenizer is loaded before attempting to use it for tokenization and prediction.



import torch
from transformers import AutoTokenizer

# Define the directory where the retrained model and tokenizer were saved
save_directory = "./my_legal_bert_model"

# Load the tokenizer again to ensure it's available
try:
    loaded_tokenizer = AutoTokenizer.from_pretrained(save_directory)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer from {save_directory}: {e}")
    loaded_tokenizer = None # Ensure tokenizer is None if loading fails

# Assuming 'clauses' list and 'trainer_binary' are available from previous steps
if 'clauses' in locals() and 'trainer_binary' in locals() and trainer_binary.model is not None and loaded_tokenizer is not None:
    # 1. Tokenize the list of extracted clauses
    clause_encodings = loaded_tokenizer(clauses, truncation=True, padding=True, max_length=256, return_tensors="pt")

    # 2. Create a PyTorch dataset from the tokenized clauses
    # Use the PredictionDataset class defined in a previous step (assuming it's still in scope)
    class PredictionDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            return item

    clause_dataset = PredictionDataset(clause_encodings)

    # 3. Use the trainer_binary.predict() method to get predictions (logits) for the clause dataset
    print("\nMaking predictions on extracted clauses...")
    # Move the model to the correct device if it's not already
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer_binary.model.to(device)

    predictions_clauses = trainer_binary.predict(clause_dataset)

    # 4. Convert the predicted logits into predicted class indices (0 or 1)
    predicted_logits_clauses = predictions_clauses.predictions
    predicted_indices_clauses = predicted_logits_clauses.argmax(axis=1)

    # 5. Store the predicted binary labels for each clause
    # Convert to a Python list or numpy array if needed for further processing
    predicted_binary_labels_per_clause = predicted_indices_clauses.tolist()

    print("Predictions completed.")
    print(f"Predicted binary labels for {len(clauses)} clauses: {predicted_binary_labels_per_clause}")

    # Optionally, print the clauses with their predicted labels
    print("\nClauses with Predicted Binary Labels:")
    binary_label_map = {0: 'Real (0)', 1: 'Fake (1)'}
    for i, (clause, label_index) in enumerate(zip(clauses, predicted_binary_labels_per_clause)):
        label_text = binary_label_map.get(label_index, "Unknown")
        print(f"Clause {i+1}: {clause}")
        print(f"Predicted Label: {label_text}\n")

else:
    print("Error: Clauses, trainer_binary, or loaded_tokenizer not found. Cannot make predictions.")
    predicted_binary_labels_per_clause = None


**Reasoning**:
The previous attempt to load the tokenizer failed with a `Repo id` error, which indicates `AutoTokenizer.from_pretrained` was likely trying to interpret the local path as a Hugging Face Hub repository ID. When loading from a local directory, the path should be provided directly. I need to correct the loading mechanism for the tokenizer and model. Also, I need to load the model explicitly as `trainer_binary` was not found in the previous state.



import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define the directory where the retrained model and tokenizer were saved
save_directory = "./my_legal_bert_model"

# Load the tokenizer directly from the local directory
try:
    loaded_tokenizer = AutoTokenizer.from_pretrained(save_directory)
    print("Tokenizer loaded successfully from local directory.")
except Exception as e:
    print(f"Error loading tokenizer from local directory {save_directory}: {e}")
    loaded_tokenizer = None # Ensure tokenizer is None if loading fails

# Load the model directly from the local directory
try:
    loaded_model_binary = AutoModelForSequenceClassification.from_pretrained(save_directory)
    print("Model loaded successfully from local directory.")
except Exception as e:
    print(f"Error loading model from local directory {save_directory}: {e}")
    loaded_model_binary = None # Ensure model is None if loading fails


# Assuming 'clauses' list is available from previous steps
if 'clauses' in locals() and loaded_model_binary is not None and loaded_tokenizer is not None:
    # 1. Tokenize the list of extracted clauses
    clause_encodings = loaded_tokenizer(clauses, truncation=True, padding=True, max_length=256, return_tensors="pt")

    # 2. Create a PyTorch dataset from the tokenized clauses
    # Use the PredictionDataset class defined in a previous step (assuming it's still in scope)
    class PredictionDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            return item

    clause_dataset = PredictionDataset(clause_encodings)

    # 3. Make predictions using the loaded model
    print("\nMaking predictions on extracted clauses...")
    # Move the model to the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model_binary.to(device)
    loaded_model_binary.eval() # Set model to evaluation mode

    # Use a DataLoader for batching if needed, but for a few clauses, direct prediction is fine
    # Convert dataset to tensors and move to device
    input_ids = clause_encodings['input_ids'].to(device)
    attention_mask = clause_encodings['attention_mask'].to(device)
    token_type_ids = clause_encodings['token_type_ids'].to(device)


    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = loaded_model_binary(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.logits

    # 4. Convert the predicted logits into predicted class indices (0 or 1)
    predicted_indices_clauses = torch.argmax(logits, axis=1).cpu().numpy() # Move to CPU for numpy conversion

    # 5. Store the predicted binary labels for each clause
    predicted_binary_labels_per_clause = predicted_indices_clauses.tolist()

    print("Predictions completed.")
    print(f"Predicted binary labels for {len(clauses)} clauses: {predicted_binary_labels_per_clause}")

    # Optionally, print the clauses with their predicted labels
    print("\nClauses with Predicted Binary Labels:")
    binary_label_map = {0: 'Real (0)', 1: 'Fake (1)'}
    for i, (clause, label_index) in enumerate(zip(clauses, predicted_binary_labels_per_clause)):
        label_text = binary_label_map.get(label_index, "Unknown")
        print(f"Clause {i+1}: {clause}")
        print(f"Predicted Label: {label_text}\n")

else:
    print("Error: Clauses, loaded_model_binary, or loaded_tokenizer not found. Cannot make predictions.")
    predicted_binary_labels_per_clause = None # Ensure variable is defined even on error


**Reasoning**:
The error message "Repo id must use alphanumeric chars or '-', '_', '.', '--' and '..' are forbidden, '-' and '.' cannot start or end the name, max length is 96: './my_legal_bert_model'" indicates that `AutoTokenizer.from_pretrained` and `AutoModelForSequenceClassification.from_pretrained` are still trying to parse the local path `./my_legal_bert_model` as a Hugging Face Hub repository ID. This is unexpected behavior when providing a local path. A potential workaround is to use the `from_pretrained` method with the full absolute path or ensure the path is correctly interpreted as a local directory. Let's try providing the absolute path explicitly. Also, ensure the `clauses` list is available.



import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os # Import os to get absolute path

# Define the directory where the retrained model and tokenizer were saved
save_directory = "./my_legal_bert_model"
abs_save_directory = os.path.abspath(save_directory) # Get the absolute path

# Load the tokenizer directly from the local directory using the absolute path
try:
    loaded_tokenizer = AutoTokenizer.from_pretrained(abs_save_directory)
    print("Tokenizer loaded successfully from local directory using absolute path.")
except Exception as e:
    print(f"Error loading tokenizer from local directory {abs_save_directory}: {e}")
    loaded_tokenizer = None # Ensure tokenizer is None if loading fails

# Load the model directly from the local directory using the absolute path
try:
    loaded_model_binary = AutoModelForSequenceClassification.from_pretrained(abs_save_directory)
    print("Model loaded successfully from local directory using absolute path.")
except Exception as e:
    print(f"Error loading model from local directory {abs_save_directory}: {e}")
    loaded_model_binary = None # Ensure model is None if loading fails


# Assuming 'clauses' list is available from previous steps
if 'clauses' in locals() and loaded_model_binary is not None and loaded_tokenizer is not None:
    # 1. Tokenize the list of extracted clauses
    clause_encodings = loaded_tokenizer(clauses, truncation=True, padding=True, max_length=256, return_tensors="pt")

    # 2. Create a PyTorch dataset from the tokenized clauses
    # Use the PredictionDataset class defined in a previous step (assuming it's still in scope)
    class PredictionDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            return item

    clause_dataset = PredictionDataset(clause_encodings)

    # 3. Make predictions using the loaded model
    print("\nMaking predictions on extracted clauses...")
    # Move the model to the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model_binary.to(device)
    loaded_model_binary.eval() # Set model to evaluation mode

    # Convert dataset to tensors and move to device
    input_ids = clause_encodings['input_ids'].to(device)
    attention_mask = clause_encodings['attention_mask'].to(device)
    token_type_ids = clause_encodings['token_type_ids'].to(device)


    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = loaded_model_binary(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.logits

    # 4. Convert the predicted logits into predicted class indices (0 or 1)
    predicted_indices_clauses = torch.argmax(logits, axis=1).cpu().numpy() # Move to CPU for numpy conversion

    # 5. Store the predicted binary labels for each clause
    predicted_binary_labels_per_clause = predicted_indices_clauses.tolist()

    print("Predictions completed.")
    print(f"Predicted binary labels for {len(clauses)} clauses: {predicted_binary_labels_per_clause}")

    # Optionally, print the clauses with their predicted labels
    print("\nClauses with Predicted Binary Labels:")
    binary_label_map = {0: 'Real (0)', 1: 'Fake (1)'}
    for i, (clause, label_index) in enumerate(zip(clauses, predicted_binary_labels_per_clause)):
        label_text = binary_label_map.get(label_index, "Unknown")
        print(f"Clause {i+1}: {clause}")
        print(f"Predicted Label: {label_text}\n")

else:
    print("Error: Clauses, loaded_model_binary, or loaded_tokenizer not found. Cannot make predictions.")
    predicted_binary_labels_per_clause = None # Ensure variable is defined even on error


**Reasoning**:
The `from_pretrained` method is still trying to interpret the local path as a Hugging Face Hub repository ID, even with the absolute path. This is unusual. A more explicit way to load from a local directory might be needed, or there might be an environment issue causing this behavior. Since the task is to apply the model, and the primary issue is loading the model and tokenizer, I will try a different approach to loading the model and tokenizer that is less likely to be confused with a Hugging Face Hub ID, and then proceed with tokenization and prediction as planned.



import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Define the directory where the retrained model and tokenizer were saved
save_directory = "./my_legal_bert_model"
abs_save_directory = os.path.abspath(save_directory)

# Load the tokenizer directly from the local directory
try:
    # Try a more explicit way to load from a local path
    loaded_tokenizer = AutoTokenizer.from_pretrained(abs_save_directory)
    print("Tokenizer loaded successfully from local directory.")
except Exception as e:
    print(f"Error loading tokenizer from local directory {abs_save_directory}: {e}")
    loaded_tokenizer = None # Ensure tokenizer is None if loading fails

# Load the model directly from the local directory
try:
    # Try a more explicit way to load from a local path
    loaded_model_binary = AutoModelForSequenceClassification.from_pretrained(abs_save_directory)
    print("Model loaded successfully from local directory.")
except Exception as e:
    print(f"Error loading model from local directory {abs_save_directory}: {e}")
    loaded_model_binary = None # Ensure model is None if loading fails


# Assuming 'clauses' list is available from previous steps
if 'clauses' in locals() and loaded_model_binary is not None and loaded_tokenizer is not None:
    # 1. Tokenize the list of extracted clauses
    clause_encodings = loaded_tokenizer(clauses, truncation=True, padding=True, max_length=256, return_tensors="pt")

    # 2. Create a PyTorch dataset from the tokenized clauses
    # Use the PredictionDataset class defined in a previous step (assuming it's still in scope)
    class PredictionDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            return item

    clause_dataset = PredictionDataset(clause_encodings)

    # 3. Make predictions using the loaded model
    print("\nMaking predictions on extracted clauses...")
    # Move the model to the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model_binary.to(device)
    loaded_model_binary.eval() # Set model to evaluation mode

    # Convert dataset to tensors and move to device
    input_ids = clause_encodings['input_ids'].to(device)
    attention_mask = clause_encodings['attention_mask'].to(device)
    token_type_ids = clause_encodings['token_type_ids'].to(device)


    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = loaded_model_binary(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.logits

    # 4. Convert the predicted logits into predicted class indices (0 or 1)
    predicted_indices_clauses = torch.argmax(logits, axis=1).cpu().numpy() # Move to CPU for numpy conversion

    # 5. Store the predicted binary labels for each clause
    predicted_binary_labels_per_clause = predicted_indices_clauses.tolist()

    print("Predictions completed.")
    print(f"Predicted binary labels for {len(clauses)} clauses: {predicted_binary_labels_per_clause}")

    # Optionally, print the clauses with their predicted labels
    print("\nClauses with Predicted Binary Labels:")
    binary_label_map = {0: 'Real (0)', 1: 'Fake (1)'}
    for i, (clause, label_index) in enumerate(zip(clauses, predicted_binary_labels_per_clause)):
        label_text = binary_label_map.get(label_index, "Unknown")
        print(f"Clause {i+1}: {clause}")
        print(f"Predicted Label: {label_text}\n")

else:
    print("Error: Clauses, loaded_model_binary, or loaded_tokenizer not found. Cannot make predictions.")
    predicted_binary_labels_per_clause = None # Ensure variable is defined even on error
