[[中文版](https://github.com/JuneYaooo/medical_kb_chatbot/blob/main/README.md)] [[English](https://github.com/JuneYaooo/medical_kb_chatbot/blob/main/README_en.md)]

# 医疗知识聊天机器人

欢迎使用 医疗知识聊天机器人，这是一款基于 PULSE 模型，引入知识库及微调训练的聊天机器人，旨在提供更实用的医疗相关功能和服务。用户可以自己添加相关知识库，进行模型微调，体验更丰富的应用场景：

## 示例医疗聊天机器人应用场景

- **药物查询**：提供药物数据库，用户可以搜索特定药物的信息，如用途、剂量、副作用等。

- **病症解释**：提供常见疾病、症状和医学术语的解释和定义，帮助用户更好地理解医学知识。

- **医疗客服**：添加相关医疗产品文档，支持用户与聊天机器人进行个性化对话，回答医疗产品相关问题，提供准确和可靠的信息。

## 使用方法

### 安装

首先，克隆本项目到本地计算机：

```
git clone https://github.com/JuneYaooo/medical_kb_chatbot.git
```

#### 使用 pip 安装

确保您的计算机上已安装以下依赖项：

- Python 3.8
- pip 包管理器

进入项目目录并安装必要的依赖项：

```
cd medical_kb_chatbot
pip install -r requirements.txt
```

#### 使用 conda 安装

确保您的计算机上已安装以下依赖项：

- Anaconda 或 Miniconda

进入项目目录并创建一个新的 conda 环境：

```
cd medical_kb_chatbot
conda env create -f environment.yml
```

激活新创建的环境：

```
conda activate med_llm
```

然后运行聊天机器人：

```
python backend_app.py
```

### 交互

聊天机器人将在终端上提供一个简单的交互界面。您可以根据提示输入相关信息，选择要执行的功能。

## 贡献

如果您对该项目感兴趣，欢迎贡献您的代码和改进建议。您可以通过以下方式参与：

1. 提交问题和建议到本项目的 Issue 页面。
2. Fork 本项目并提交您的改进建议，我们将会审查并合并合适的改动。
