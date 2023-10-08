from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load pre-trained GPT-2 model and tokenizer
model_name_gpt = "gpt2"  # You can change this to any other Hugging Face model
model_name_lama = "microsoft/Llama2-7b-WhoIsHarryPotter"

model_gpt = GPT2LMHeadModel.from_pretrained(model_name_gpt)
tokenizer_gpt = GPT2Tokenizer.from_pretrained(model_name_gpt)

tokenizer_lama = AutoTokenizer.from_pretrained(model_name_lama)
model_lama = AutoModelForCausalLM.from_pretrained(model_name_lama)

# Set the device to GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model_gpt.to(device)
model_lama.to(device)

while True:
    # Prompt the user for input
    user_input = input("Enter a prompt (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # GPT response
    # Encode the user's input text to tensor
    input_ids = tokenizer_gpt.encode(user_input, return_tensors="pt").to(device)
    # Generate text based on the input
    with torch.no_grad():
        output = model_gpt.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
        
    # Llama response
        input_ids = tokenizer_lama.encode(user_input, return_tensors="pt").to(device)
    # Generate text based on the input
    with torch.no_grad():
        output = model_lama.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95)

    # Decode and print the generated text
    generated_text_gpt = tokenizer_gpt.decode(output[0], skip_special_tokens=True)
    generated_text_lama = tokenizer_lama.decode(output[0], skip_special_tokens=True)
    print("Generated Text from GPT2:")
    print(generated_text_gpt)
    print("Generated Text from Llama:")
    print(generated_text_lama)
