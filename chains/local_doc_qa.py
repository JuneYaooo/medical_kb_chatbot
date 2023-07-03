from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader, TextLoader
from configs.common_config import *
import datetime
from loader.textsplitter import ChineseTextSplitter
from typing import List, Tuple, Dict
from langchain.docstore.document import Document
import numpy as np
from loader.utils import torch_gc
from tqdm import tqdm
from pypinyin import lazy_pinyin
from loader import UnstructuredPaddleImageLoader, UnstructuredPaddlePDFLoader
from loader.models.base import (BaseAnswer,
                         AnswerResult,
                         AnswerResultStream,
                         AnswerResultQueueSentinelTokenListenerQueue)
from loader.models.loader.args import parser
from loader.models.loader import LoaderCheckPoint
import loader.models.shared as shared
# from agent import bing_search
from langchain.docstore.document import Document
from .modules.json_load import JsonLoader
from .modules.excel_load import ExcelLoader

def load_file(filepath, sentence_size=SENTENCE_SIZE):
    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.lower().endswith(".txt"):
        loader = TextLoader(filepath, autodetect_encoding=True)
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".pdf"):
        loader = UnstructuredPaddlePDFLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=True, sentence_size=sentence_size)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".jpg") or filepath.lower().endswith(".png"):
        loader = UnstructuredPaddleImageLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
    elif filepath.lower().endswith(".jsonl") or filepath.lower().endswith(".json"):
        loader = JsonLoader(filepath, autodetect_encoding=True)
        docs = loader.load_json()
    elif filepath.lower().endswith(".xlsx"):
        loader = ExcelLoader(filepath, autodetect_encoding=True)
        docs = loader.load_excel()
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
    write_check_file(filepath, docs)
    return docs


def write_check_file(filepath, docs):
    folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fp = os.path.join(folder_path, 'load_file.txt')
    with open(fp, 'a+', encoding='utf-8') as fout:
        fout.write("filepath=%s,len=%s" % (filepath, len(docs)))
        fout.write('\n')
        for i in docs:
            fout.write(str(i))
            fout.write('\n')
        fout.close()

def concat_history(model_name:str,chat_history: List[str]=[],query:str = '',process_type='chat'):
    formatted_history = ""
    if process_type=='chat':
        if 'pulse' in model_name.lower():
            for sublist in chat_history:
                if sublist[0] is not None:
                    formatted_history += f"\nUser: {sublist[0]}\nHelper: {sublist[1]}"
        else:
            for sublist in chat_history:
                if sublist[0] is not None:
                    formatted_history += f"\nUser: {sublist[0]}\nHelper: {sublist[1]}"
    elif process_type=='search':
            chat_history = chat_history[-1:]
            for j,sublist in enumerate(chat_history, start=1):
                if sublist[0] is not None:
                    formatted_history += f"User:{sublist[0]}Helper: {sublist[1]}"
            formatted_history+=f"User:{query}"
    return formatted_history

def remove_duplicates(lst):
    seen = set()
    result = []
    for i, item in enumerate(lst, start=1):
        page_content = item.page_content
        if page_content not in seen:
            seen.add(page_content)
            result.append(item)
    return result

def generate_prompt(model_name:str, related_docs: List[str],
                    query: str,
                    chat_history: List[str]=[],prompt_template: str = PROMPT_TEMPLATE,) -> str:
    related_docs = remove_duplicates(related_docs)
    if 'pulse' in model_name.lower():
            formatted_context = ""
            for i, doc in enumerate(related_docs, start=1):
                formatted_context += f"[{i}]\n```\n{doc.page_content}\n```\n"
            formatted_context = "None" if formatted_context == "" else formatted_context
            formatted_history = concat_history(model_name,chat_history,query,process_type='chat')
            prompt = prompt_template.replace("{question}", query).replace("{context}", formatted_context).replace("{chat_history}", formatted_history)
    else:
        if len(related_docs)>0:
            formatted_context = ""
            for i, doc in enumerate(related_docs, start=1):
                formatted_context += f"[{i}]{doc.page_content}\n"
            formatted_context = "None" if formatted_context == "" else formatted_context
            formatted_history = concat_history(model_name,chat_history,query,process_type='chat')
            prompt = prompt_template.replace("{question}", query).replace("{context}", formatted_context).replace("{chat_history}", formatted_history)
        else:
            prompt = prompt_template.replace("{question}", query)
    return prompt


def seperate_list(ls: List[int]) -> List[List[int]]:
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i - 1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists


def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4
) -> List[Tuple[Document, float]]:
    scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
    docs = []
    id_set = set()
    store_len = len(self.index_to_docstore_id)
    for j, i in enumerate(indices[0]):
        if i == -1 or 0 < self.score_threshold < scores[0][j]:
            # This happens when not enough docs are returned.
            continue
        _id = self.index_to_docstore_id[i]
        doc = self.docstore.search(_id)
        if not self.chunk_content:
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            doc.metadata["score"] = int(scores[0][j])
            docs.append(doc)
            continue
        id_set.add(i)
        docs_len = len(doc.page_content)
        for k in range(1, max(i, store_len - i)):
            break_flag = False
            for l in [i + k, i - k]:
                if 0 <= l < len(self.index_to_docstore_id):
                    _id0 = self.index_to_docstore_id[l]
                    doc0 = self.docstore.search(_id0)
                    if docs_len + len(doc0.page_content) > self.chunk_size:
                        break_flag = True
                        break
                    elif doc0.metadata["source"] == doc.metadata["source"]:
                        docs_len += len(doc0.page_content)
                        id_set.add(l)
            if break_flag:
                break
    if not self.chunk_content:
        return docs
    if len(id_set) == 0 and self.score_threshold > 0:
        return []
    id_list = sorted(list(id_set))
    id_lists = seperate_list(id_list)
    for id_seq in id_lists:
        for id in id_seq:
            if id == id_seq[0]:
                _id = self.index_to_docstore_id[id]
                doc = self.docstore.search(_id)
            else:
                _id0 = self.index_to_docstore_id[id]
                doc0 = self.docstore.search(_id0)
                doc.page_content += " " + doc0.page_content
        if not isinstance(doc, Document):
            raise ValueError(f"Could not find document for id {_id}, got {doc}")
        doc_score = min([scores[0][id] for id in [indices[0].tolist().index(i) for i in id_seq if i in indices[0]]])
        doc.metadata["score"] = int(doc_score)
        docs.append(doc)
    torch_gc()
    return docs


def search_result2docs(search_results):
    docs = []
    for result in search_results:
        doc = Document(page_content=result["snippet"] if "snippet" in result.keys() else "",
                       metadata={"source": result["link"] if "link" in result.keys() else "",
                                 "filename": result["title"] if "title" in result.keys() else ""})
        docs.append(doc)
    return docs


class LocalDocQA:
    llm: BaseAnswer = None
    embeddings: object = None
    top_k: int = VECTOR_SEARCH_TOP_K
    chunk_size: int = CHUNK_SIZE
    chunk_content: bool = True
    score_threshold: int = VECTOR_SEARCH_SCORE_THRESHOLD

    def init_cfg(self,
                 embedding_model: str = EMBEDDING_MODEL,
                 embedding_device=EMBEDDING_DEVICE,
                 llm_model: BaseAnswer = None,
                 top_k=VECTOR_SEARCH_TOP_K,
                 ):
        self.llm = llm_model
        print('embedding_device',embedding_device)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model],
                                                model_kwargs={'device': embedding_device})
        self.top_k = top_k

    def init_embedding(self,
                 embedding_model: str = EMBEDDING_MODEL,
                 embedding_device=EMBEDDING_DEVICE,
                 top_k=VECTOR_SEARCH_TOP_K,
                 ):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model],
                                                model_kwargs={'device': embedding_device})
        self.top_k = top_k

    def init_knowledge_vector_store(self,
                                    filepath: str or List[str],
                                    vs_path: str or os.PathLike = None,
                                    sentence_size=SENTENCE_SIZE):
        loaded_files = []
        failed_files = []
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                print("路径不存在")
                return None
            elif os.path.isfile(filepath):
                file = os.path.split(filepath)[-1]
                try:
                    docs = load_file(filepath, sentence_size)
                    logger.info(f"{file} 已成功加载")
                    loaded_files.append(filepath)
                except Exception as e:
                    logger.error(e)
                    logger.info(f"{file} 未能成功加载")
                    return None
            elif os.path.isdir(filepath):
                docs = []
                for file in tqdm(os.listdir(filepath), desc="加载文件"):
                    fullfilepath = os.path.join(filepath, file)
                    try:
                        docs += load_file(fullfilepath, sentence_size)
                        loaded_files.append(fullfilepath)
                    except Exception as e:
                        logger.error(e)
                        failed_files.append(file)

                if len(failed_files) > 0:
                    logger.info("以下文件未能成功加载：")
                    for file in failed_files:
                        logger.info(f"{file}\n")

        else:
            docs = []
            for file in filepath:
                try:
                    docs += load_file(file)
                    logger.info(f"{file} 已成功加载")
                    loaded_files.append(file)
                except Exception as e:
                    logger.error(e)
                    logger.info(f"{file} 未能成功加载")
        if len(docs) > 0:
            logger.info("文件加载完毕，正在生成向量库")
            if vs_path and os.path.isdir(vs_path):
                vector_store = FAISS.load_local(vs_path, self.embeddings)
                vector_store.add_documents(docs)
                torch_gc()
            else:
                if not vs_path:
                    vs_path = os.path.join(VS_ROOT_PATH,
                                           f"""{"".join(lazy_pinyin(os.path.splitext(file)[0]))}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}""")
                vector_store = FAISS.from_documents(docs, self.embeddings)  # docs 为Document列表
                torch_gc()

            vector_store.save_local(vs_path)
            return vs_path, loaded_files
        else:
            logger.info("文件均未成功加载，请检查依赖包或替换为其他文件再次上传。")
            return None, loaded_files

    def one_knowledge_add(self, vs_path, one_title, one_conent, one_content_segmentation, sentence_size):
        try:
            if not vs_path or not one_title or not one_conent:
                logger.info("知识库添加错误，请确认知识库名字、标题、内容是否正确！")
                return None, [one_title]
            docs = [Document(page_content=one_conent + "\n", metadata={"source": one_title})]
            if not one_content_segmentation:
                text_splitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
                docs = text_splitter.split_documents(docs)
            if os.path.isdir(vs_path):
                vector_store = FAISS.load_local(vs_path, self.embeddings)
                vector_store.add_documents(docs)
            else:
                vector_store = FAISS.from_documents(docs, self.embeddings)  ##docs 为Document列表
            torch_gc()
            vector_store.save_local(vs_path)
            return vs_path, [one_title]
        except Exception as e:
            logger.error(e)
            return None, [one_title]

    def get_knowledge_based_answer(self, model_name, query, vs_path, prompt_template, chat_history=[], streaming: bool = STREAMING):
        vector_store = FAISS.load_local(vs_path, self.embeddings)
        FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        vector_store.chunk_size = self.chunk_size
        vector_store.chunk_content = self.chunk_content
        vector_store.score_threshold = self.score_threshold
        formatted_history = concat_history(model_name,chat_history,query,process_type='search')
        # 根据历史记录来检索
        related_docs_with_score = vector_store.similarity_search_with_score(formatted_history, k=self.top_k)
        torch_gc()
        prompt = generate_prompt(model_name, related_docs_with_score, query, chat_history, prompt_template)

        for answer_result in self.llm.generatorAnswer(prompt=prompt, history=chat_history,
                                                      streaming=streaming):
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][0] = query
            response = {"query": query,
                        "result": resp,
                        "source_documents": related_docs_with_score}
            yield response, history

    def get_base_answer(self, model_name, query, prompt_template, chat_history=[], streaming: bool = STREAMING):
        formatted_history = concat_history(model_name,chat_history,query,process_type='chat')
        torch_gc()
        if prompt_template == '':
            prompt = f"{formatted_history}</s>User:{query}</s>Helper:"
        else:
            prompt = generate_prompt(model_name, [], query, chat_history, prompt_template)

        for answer_result in self.llm.generatorAnswer(prompt=prompt, history=chat_history,
                                                    streaming=streaming):
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][0] = query
            response = {"query": query,
                        "result": resp}
            yield response, history

    # query      查询内容
    # vs_path    知识库路径
    # chunk_content   是否启用上下文关联
    # score_threshold    搜索匹配score阈值
    # vector_search_top_k   搜索知识库内容条数，默认搜索5条结果
    # chunk_sizes    匹配单段内容的连接上下文长度
    def get_knowledge_based_conent_test(self, query, vs_path, chunk_content,
                                        score_threshold=VECTOR_SEARCH_SCORE_THRESHOLD,
                                        vector_search_top_k=VECTOR_SEARCH_TOP_K, chunk_size=CHUNK_SIZE):
        vector_store = FAISS.load_local(vs_path, self.embeddings)
        FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        vector_store.chunk_content = chunk_content
        vector_store.score_threshold = score_threshold
        vector_store.chunk_size = chunk_size
        related_docs_with_score = vector_store.similarity_search_with_score(query, k=vector_search_top_k)
        if not related_docs_with_score:
            response = {"query": query,
                        "source_documents": []}
            return response, ""
        torch_gc()
        prompt = "\n".join([doc.page_content for doc in related_docs_with_score])
        response = {"query": query,
                    "source_documents": related_docs_with_score}
        return response, prompt

    def get_search_result_based_answer(self,model_name, query, chat_history=[],prompt_template=BASE_PROMPT_TEMPLATE, streaming: bool = STREAMING):
        results = bing_search(query)
        result_docs = search_result2docs(results)
        prompt = generate_prompt(model_name, result_docs, query,  chat_history, prompt_template)

        for answer_result in self.llm.generatorAnswer(prompt=prompt, history=chat_history,
                                                      streaming=streaming):
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][0] = query
            response = {"query": query,
                        "result": resp,
                        "source_documents": result_docs}
            yield response, history


