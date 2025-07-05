# 📝 Story Generator using Generative AI

An AI-powered story generator that creates coherent and engaging narratives using both fine-tuned and pretrained language models. This project demonstrates the capabilities of generative AI in creative writing through an interactive web interface.

## 🚀 Features

- **Dual Model Architecture**: 
  - Fine-tuned GPT-2 model for narrative generation
  - Llama-3.2-1B-Instruct via Ollama for structured outputs (poetry, short stories)
- **Interactive Web Interface**: User-friendly Gradio GUI with adjustable parameters
- **Customizable Generation**: Control creativity, length, and style through sliders
- **GPU Acceleration**: Optimized for CUDA-enabled devices with CPU fallback
- **Real-time Generation**: Fast inference with response times under 15 seconds

## 📋 Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Configuration](#configuration)
- [Examples](#examples)
- [Testing](#testing)
- [Contributing](#contributing)
- [Future Enhancements](#future-enhancements)
- [License](#license)

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster inference)
- At least 8GB RAM
- Internet connection for model downloads

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Adarsh2059/Story_Generator_using_Gen_AI.git
   cd Story_Generator_using_Gen_AI
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install transformers datasets torch
   pip install ollama
   pip install gradio
   ```

4. **Install Ollama and download the model**
   ```bash
   # Install Ollama (visit https://ollama.com for OS-specific instructions)
   ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
   ```

## 🎮 Usage

### Training the Fine-tuned Model

1. **Prepare your dataset**
   - Place your stories/text data in `Stories.txt`
   - Ensure each story is separated by double newlines (`\n\n`)

2. **Run the fine-tuning script**
   ```python
   python fine_tune_gpt2.py
   ```

### Running the Story Generator

1. **For GPT-2 Fine-tuned Model**
   ```python
   python gpt2_generator.py
   ```

2. **For Ollama-based Generation**
   ```python
   python ollama_generator.py
   ```

3. **Launch the Web Interface**
   ```python
   python gradio_app.py
   ```
   
   The interface will be available at `http://localhost:7860`

### Quick Start Example

```python
from transformers import pipeline

# Load the fine-tuned model
generator = pipeline("text-generation", model="./gpt2-finetuned-prototype")

# Generate a story
prompt = "Once upon a time in a magical forest"
story = generator(prompt, max_length=150, temperature=0.9)
print(story[0]['generated_text'])
```

## 📁 Project Structure

```
Story_Generator_using_Gen_AI/
├── README.md
├── requirements.txt
├── Stories.txt                 # Training dataset
├── code/
│   ├── fine_tune_gpt2.py      # GPT-2 fine-tuning script
│   ├── gpt2_generator.py      # GPT-2 story generation
│   ├── ollama_generator.py    # Ollama-based generation
│   └── gradio_app.py          # Web interface
├── models/
│   └── gpt2-finetuned-prototype/  # Fine-tuned model directory
├── examples/
│   └── sample_outputs.txt     # Example generated stories
└── docs/
    └── technical_report.pdf   # Detailed project documentation
```

## 🤖 Model Details

### Fine-tuned GPT-2 Model

- **Base Model**: GPT-2 (124M parameters)
- **Training**: Custom dataset of stories
- **Optimization**: 
  - Learning rate: 5e-5
  - Batch size: 16
  - Max length: 110 tokens
  - Mixed precision training (FP16)

### Llama-3.2-1B-Instruct

- **Access**: Via Ollama API
- **Specialization**: Structured outputs (poetry, short stories)
- **Parameters**: 1 billion parameters
- **Format**: GGUF (optimized for local inference)

## ⚙️ Configuration

### Generation Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `temperature` | 0.1-2.0 | 0.9 | Controls randomness (higher = more creative) |
| `max_length` | 50-300 | 150 | Maximum tokens to generate |
| `top_k` | 10-100 | 50 | Limits vocabulary for next token |
| `top_p` | 0.1-1.0 | 0.95 | Nucleus sampling threshold |

### Training Configuration

```python
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-prototype",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    learning_rate=5e-5,
    warmup_steps=10,
    max_steps=100,
    fp16=True,
    optim="adamw_torch"
)
```

## 📚 Examples

### Story Generation

**Input**: "The ancient wizard opened the dusty tome"

**Output**: "The ancient wizard opened the dusty tome, and golden light spilled forth like liquid sunshine. The pages whispered secrets of forgotten spells, each word pulsing with magical energy that made the air itself shimmer with possibility."

### Poetry Generation

**Input**: "Write a poem about a tiger in the woods"

**Output**: 
```
Stripes of shadow, stripes of flame,
Through morning mist, the tiger came.
Padding soft on forest floor,
Ancient wisdom in his core.
```

## 🧪 Testing

### Run Test Suite

```bash
python -m pytest tests/
```

### Manual Testing

1. **Input Validation**: Test with various prompt types
2. **Parameter Sensitivity**: Experiment with different temperature values
3. **Edge Cases**: Try unusual or incomplete prompts
4. **Performance**: Monitor generation times and memory usage

### Test Results

- **Coherence Rate**: 85% of outputs rated as contextually appropriate
- **Average Response Time**: 2-5 seconds (GPU), 6-15 seconds (CPU)
- **Edge Case Handling**: Robust error handling with graceful degradation

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🔮 Future Enhancements

- [ ] **Multi-modal Support**: Image-to-text story generation
- [ ] **Genre Specialization**: Fine-tuned models for specific genres
- [ ] **Collaborative Writing**: Real-time co-creation features
- [ ] **Multilingual Support**: Story generation in multiple languages
- [ ] **Voice Integration**: Audio input/output capabilities
- [ ] **Export Options**: PDF, EPUB, and other formats
- [ ] **Advanced Editing**: In-line editing and refinement tools

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Model Size | 124M parameters (GPT-2) |
| Training Time | ~30 minutes (GTX 1060) |
| Inference Speed | 2-5 seconds (GPU) |
| Memory Usage | ~2GB (GPU inference) |
| Coherence Score | 85% |


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for the Transformers library
- OpenAI for the GPT-2 model
- Ollama team for local LLM deployment
- Gradio team for the web interface framework

## 📧 Contact

**Adarsh Yadav**
- Email: adarsh.23bce10598@vitbhopal.ac.in
- GitHub: [@Adarsh2059](https://github.com/Adarsh2059)
- University: VIT Bhopal University

---

⭐ If you found this project helpful, please consider giving it a star!

---
