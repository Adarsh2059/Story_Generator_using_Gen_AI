import ollama

def generate_poetry(prompt, model="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF",max_tokens = 200):
    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={
            "temperature": 0.8,     
            "num_predict": max_tokens       
        }
    )
    return response['response']

prompt = ("Write a short, evocative English poem about a tiger going for a morning walk in the woods.")
poem = generate_poetry(prompt)
print("\n--- Generated Poem ---\n")
print(poem)
