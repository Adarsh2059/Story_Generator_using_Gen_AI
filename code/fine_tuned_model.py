import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Load your fine-tuned model and tokenizer
model_path = r"C:\Users\lenovo\Documents\GitHub\Story_Generator_using_Gen_AI\code\gpt2-finetuned-prototype"
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Using original GPT-2 tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path)

# Set pad token (important for generation)
tokenizer.pad_token = tokenizer.eos_token

# Create text generation pipeline
story_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1  # Use GPU if available
)

# Generate text
prompt = "In the starting of the week,"  # Your starting prompt
generated_stories = story_generator(
    prompt,
    max_length=150,  # Maximum length of generated text
    num_return_sequences=3,  # Number of different stories to generate
    temperature=0.7,  # Controls randomness (lower = more predictable)
    top_k=50,  # Limits to top 50 probable next words
    do_sample=True,  # Enables random sampling
)

# Print results
for i, story in enumerate(generated_stories):
    print(f"\nStory {i+1}:")
    print(story['generated_text'])
    print("-" * 50)