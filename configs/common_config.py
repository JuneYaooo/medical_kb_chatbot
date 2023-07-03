import torch.cuda
import torch.backends
import os
import logging
import uuid
import datetime

def get_formatted_date():
    current_date = datetime.date.today()
    formatted_date = current_date.strftime("%Y-%m-%d")
    return formatted_date

date_string = get_formatted_date()

LOG_FORMAT = "%(levelname) -5s %(asctime)s" "-1d: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)

embedding_model_dict = {
        "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
        "ernie-base": "nghuyong/ernie-3.0-base-zh",
        "text2vec-base": "shibing624/text2vec-base-chinese",
        "text2vec-large-chinese": "GanymedeNil/text2vec-large-chinese",
        "m3e-small": "moka-ai/m3e-small",
        "m3e-base": "moka-ai/m3e-base",
}

# Embedding model name
EMBEDDING_MODEL = "text2vec-large-chinese"

# Embedding running device
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# supported LLM models
# llm_model_dict 处理了loader的一些预设行为，如加载位置，模型名称，模型处理器实例
llm_model_dict = {
    "PULSE": {
                    "name": "PULSE",
                    "pretrained_model_name": "OpenMEDLab/PULSE-7bv5",
                    "local_model_path": "OpenMEDLab/PULSE-7bv5",
                    "provides": "Bloomz"
                }
}

# LLM 名称
LLM_MODEL = "PULSE"
# 如果你需要加载本地的model，指定这个参数  ` --no-remote-model`，或者下方参数修改为 `True`
NO_REMOTE_MODEL = False
# 量化加载8bit 模型
LOAD_IN_8BIT = False
# Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.
BF16 = False
# 本地模型存放的位置
MODEL_DIR = "model/"
# 本地lora存放的位置
LORA_DIR = "loras/"

# LLM lora path，默认为空，如果有请直接指定文件夹路径
LLM_LORA_PATH = ""
USE_LORA = True if LLM_LORA_PATH else False

# LLM streaming reponse
STREAMING = False

# Use p-tuning-v2 PrefixEncoder
USE_PTUNING_V2 = False

# LLM running device
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


VS_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store")

UPLOAD_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "content")

# 基于上下文的prompt模版，请务必保留"{question}"和"{context}"
PROMPT_TEMPLATE = """已知信息：
{context} 

假设你是客服，根据上述已知信息，来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""

BASE_PROMPT_TEMPLATE = """Instruction: 假如你是机器人。Input:{question}"""


# 文本分句长度
SENTENCE_SIZE = 20

# 匹配后单段上下文长度
CHUNK_SIZE = 250

# LLM input history length
LLM_HISTORY_LEN = 8

# return top-k text chunk from vector store
VECTOR_SEARCH_TOP_K = 3

# 知识检索内容相关度 Score, 数值范围约为0-1100，如果为0，则不生效，经测试设置为小于500时，匹配结果更精准
VECTOR_SEARCH_SCORE_THRESHOLD = 700

NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")

FLAG_USER_NAME = uuid.uuid4().hex

logger.info(f"""
loading model config
llm device: {LLM_DEVICE}
embedding device: {EMBEDDING_DEVICE}
dir: {os.path.dirname(os.path.dirname(__file__))}
flagging username: {FLAG_USER_NAME}
""")

