from llama_cpp import Llama
import os

print("Downloading and loading the model... (This may take a few minutes on the first run)")
print("Model: Qwen2.5-1.5B (Small, fast, and good for local testing)")

# We use Llama.from_pretrained to automatically download the model from Hugging Face
# The Qwen2.5-1.5B model is around 1 GB, so it should download much faster than Llama-2-7B
llm = Llama.from_pretrained(
    repo_id="bartowski/Qwen2.5-1.5B-Instruct-GGUF",
    filename="Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=4,
    verbose=False # Set to True if you want to see detailed loading logs
)

print("\nModel ready! You can now chat with the AI.")
print("(Type 'q' to quit)")

# Chat loop
while True:
    question = input("\nYou: ")
    if question.lower() == 'q':
        print("Goodbye!")
        break
    
    # Format the prompt for the Qwen model
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    # Generate response
    output = llm(
        prompt,
        max_tokens=200,
        stop=["<|im_end|>", "\nYou:"],
        echo=False
    )
    
    response = output['choices'][0]['text'].strip()
    print(f"AI: {response}")