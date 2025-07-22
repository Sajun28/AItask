# STEP 1: Install Required Packages (if not already installed)
import subprocess
import sys

def install_packages():
    try:
        import transformers
        import torch
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch"])

install_packages()

# STEP 2: Import Required Libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import warnings
warnings.filterwarnings("ignore")  # Optional: suppress all warnings

# STEP 3: Load GPT-2 Tokenizer and Model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# STEP 4: Define Text Generation Function
def generate_text(prompt, max_length=150):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones_like(inputs)

    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,            # ✅ set attention mask
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id       # ✅ set pad token
        # Removed early_stopping and num_beams to avoid warnings
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# STEP 5: Prompt & Generate
if __name__ == "__main__":
    user_prompt = input("Enter a topic for paragraph generation: ")
    print("\n--- Generated Paragraph ---\n")
    output = generate_text(user_prompt)
    print(output)
