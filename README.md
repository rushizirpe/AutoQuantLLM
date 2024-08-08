

# 🎉 Auto Quantizer

Welcome to the **Auto Quantizer**! 🚀 This is your go-to tool for easily quantizing machine learning models. With support for various methods, types, and sources, this quantizer helps you optimize your models for deployment or improve their inference speed—no hassle involved!

## ✨ Features

- **Multiple Quantization Methods**: Choose from a variety of methods, including GGUF, to fit your needs.
- **Flexible Bit Widths**: Quantize your models to different bit widths (4, 8, or 16 bits) for the best performance.
- **Source Compatibility**: Works with models from different sources, including the Hugging Face Model Hub.
- **User-Friendly Interface**: A simple command-line interface makes quantization a breeze!

## 🚀 Getting Started

### Prerequisites

Before you get started, make sure you have the following installed:

- **Python 3.7+**



### 📦 Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/rushizirpe/AutoQuantLLM.git
cd AutoQuantLLM
```

You can install the Setup by:

```bash
pip install -e .
```

### 🎈 Usage

To quantize a model, run the following command:

```bash
autoquant --model <MODEL_NAME> --method <METHOD> --bits <NUM_BITS> --output <OUTPUT_DIR> --verbose
```

- `<MODEL_NAME>` - Hugging Face model identifier (e.g., `openai-community/gpt2`).
- `<METHOD>` - quantization method you wish to use (e.g., `gguf`, `awq`).
- `<NUM_BITS>` - bit width (4, 8, 16, or 32).
- `<OUTPUT_DIR>` - directory where you want to save the quantized model.

#### 📘 Example

Here’s a quick example of how to quantize the GPT-2 model to 8 bits using the GGUF method:

```bash
autoquant --model openai-community/gpt2 --method gguf --bits 8 --output ./GGUF --verbose
```
or Qwen2-0.5B using AWQ method:

```bash
autoquant --model Qwen/Qwen2-0.5B --method awq  --bits 8 --output "awq_test" --group_size 128 --version "GEMM" --zero_point
```

### 🛠️ Supported Methods and Types

This auto quantizer supports various quantization methods and types, including:

- **GGUF**: GPT-Generated Unified Format Quantization.
- **AWQ**: Activation-aware Weight Quantization
- **Static**: To Be Added
- **Dynamic**: To Be Added
- **Weight Only**: To Be Added
- **Other methods**: To Be Added

### 🤖 Model Support

The Auto Quantizer supports models that are compatible with the Hugging Face Transformers library and other sources. Explore a wide variety of models in the [Hugging Face Model Hub](https://huggingface.co/models).

### 💌 Contributing

We’d love your contributions! If you have ideas for improvements, bug reports, or feature requests, please open an issue or submit a pull request. Your help makes this project better!

### 📄 License

This project is licensed under the MIT License. For more details, check out the [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

A big thank you to the Hugging Face team for their amazing Transformers library! And a special shoutout to the open-source community for all their support and contributions!

---
