from setuptools import setup, find_packages

setup(
    name="AutoQuantLLM",
    version="0.2",
    packages=find_packages(),
    install_requires=[
        "huggingface_hub",
        "transformers",
        "torch",
        "gitpython",
        "python-dotenv",
        "autoawq",
        "sentencepiece"
    ],
    entry_points={
        "console_scripts": [
            "autoquant=src.main:main",
        ],
    },
)