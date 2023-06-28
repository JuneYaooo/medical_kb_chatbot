[[中文版](https://github.com/JuneYaooo/medical_kb_chatbot/blob/main/README.md)] [[English](https://github.com/JuneYaooo/medical_kb_chatbot/blob/main/README_en.md)]

# Medical Knowledge Chatbot

Welcome to the Medical Knowledge Chatbot. This is a small helper based on the PULSE model, incorporating a knowledge base and fine-tuning training to provide more practical medical-related functionalities and services. Users can add relevant knowledge bases and perform model fine-tuning to experience a richer range of application scenarios.

## Examples of Medical Chatbot Use Cases

- **Drug Inquiry**: Provides a drug database where users can search for specific drug information such as usage, dosage, side effects, etc.

- **Symptom Explanation**: Provides explanations and definitions for common diseases, symptoms, and medical terminology to help users better understand medical knowledge.

- **Medical Customer Service**: Incorporates relevant medical product documentation and supports personalized conversations between users and the chatbot. It can answer questions related to medical products and provide accurate and reliable information.

## Usage

### Installation

First, clone this project to your local machine:

```
git clone https://github.com/JuneYaooo/medical_kb_chatbot.git
```

#### Installation via pip

Ensure that the following dependencies are installed on your computer:

- Python 3.8
- pip package manager

Navigate to the project directory and install the necessary dependencies:

```
cd medical_kb_chatbot
pip install -r requirements.txt
```

#### Installation via conda

Ensure that the following dependencies are installed on your computer:

- Anaconda or Miniconda

Navigate to the project directory and create a new conda environment:

```
cd medical_kb_chatbot
conda env create -f environment.yml
```

Activate the newly created environment:

```
conda activate med_llm
```

Then run the chatbot:

```
python backend_app.py
```

### Interaction

The chatbot will provide a simple interactive interface in the terminal. You can enter relevant information and select the desired functionality based on the prompts.

## Contributing

If you are interested in contributing to this project, you are welcome to contribute your code and improvement suggestions. You can participate in the following ways:

1. Submit issues and suggestions to the Issue page of this project.
2. Fork this project and submit your proposed improvements. We will review and merge appropriate changes.