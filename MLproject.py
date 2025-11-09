import torch
import torch.nn as nn
import tkinter as tk
from tkinter import scrolledtext, messagebox
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel

# --- 1. Custom Model Definition (Your Existing Code) ---
class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        # Initialize the base transformer model.
        self.model = AutoModel.from_config(config)
        # Define a classifier head.
        self.classifier = nn.Linear(config.hidden_size, 1)
        # Initialize weights (handled by PreTrainedModel)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Forward pass through the transformer
        outputs = self.model(input_ids, attention_mask=attention_mask)
        # Use the CLS token output (outputs[0][:, 0, :]) or mean pooling
        # Using mean pooling as per your original code
        last_hidden_state = outputs[0]
        
        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        # Classifier
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())

        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        return output

# --- 2. Prediction Function (Your Existing Code) ---
def predict_single_text(text, model, tokenizer, device, max_len=768, threshold=0.5):
    """
    Predicts whether the given text is AI-generated.
    """
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        probability = torch.sigmoid(logits).item()

    label = 1 if probability >= threshold else 0
    return probability, label

# --- 3. Global Initialization (for Tkinter efficiency) ---
try:
    MODEL_DIRECTORY = "desklib/ai-text-detector-v1.01"
    
    # Set up device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer and model once
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIRECTORY)
    # The model must be loaded using the custom class
    MODEL = DesklibAIDetectionModel.from_pretrained(MODEL_DIRECTORY).to(DEVICE)
    
    IS_MODEL_LOADED = True
except Exception as e:
    IS_MODEL_LOADED = False
    print(f"Error loading model or tokenizer: {e}")
    messagebox.showerror("Loading Error", "Could not load the AI Detection Model. Check the model path and dependencies.")
    # Set placeholders to prevent crash
    TOKENIZER, MODEL, DEVICE = None, None, None


# --- 4. GUI Logic (New Code) ---
def analyze_text():
    """
    Retrieves text from the ScrolledText widget and runs the AI detection logic.
    Updates the result label with the outcome.
    """
    if not IS_MODEL_LOADED:
        messagebox.showerror("Error", "Model is not loaded. Cannot analyze.")
        return
    
    input_paragraph = text_area.get("1.0", tk.END).strip()
    
    if not input_paragraph:
        messagebox.showwarning("Input Error", "Please enter a paragraph to analyze.")
        result_label.config(text="Result: Awaiting text input...")
        return

    try:
        # Run the detection function
        probability, predicted_label = predict_single_text(
            input_paragraph, MODEL, TOKENIZER, DEVICE, max_len=768, threshold=0.85
        )
        
        # Update GUI based on prediction
        if predicted_label == 1:
            result_text = f"**AI Generated** 🤖: Probability = {probability:.4f}"
            result_label.config(text=result_text, fg="red")
        else:
            result_text = f"**Not AI Generated** 🧑: Probability = {1 - probability:.4f} (Human)"
            result_label.config(text=result_text, fg="green")
            
    except Exception as e:
        messagebox.showerror("Analysis Error", f"An error occurred during prediction: {e}")
        result_label.config(text="Result: Analysis Failed.", fg="black")

# --- 5. Tkinter Setup (New Code) ---
def create_gui():
    root = tk.Tk()
    root.title("Custom AI Content Detector")
    root.geometry("700x450")

    # Title and Instructions
    tk.Label(root, text="Paste Text Below to Check for AI Generation", font=("Arial", 14, "bold")).pack(pady=10)
    
    # Scrolled Text Area for User Input
    global text_area
    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=12, font=("Arial", 11))
    text_area.pack(padx=10)

    # Analyze Button
    analyze_button = tk.Button(root, text="ANALYZE TEXT", command=analyze_text, bg="#007BFF", fg="white", font=("Arial", 12, "bold"))
    analyze_button.pack(pady=15)

    # Result Label
    global result_label
    initial_text = "Result: Model Loaded. Awaiting input..." if IS_MODEL_LOADED else "Result: MODEL LOAD ERROR!"
    result_label = tk.Label(root, text=initial_text, font=("Arial", 14), wraplength=680, justify=tk.LEFT, fg="black" if IS_MODEL_LOADED else "red")
    result_label.pack(pady=(5, 10))
    
    root.mainloop()

if __name__ == "__main__":
    # Start the GUI
    create_gui()