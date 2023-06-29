import gradio as gr
import shutil

from chains.local_doc_qa import LocalDocQA
from configs.common_config import *
import nltk
from loader.models.base import (BaseAnswer,
                         AnswerResult,
                         AnswerResultStream,
                         AnswerResultQueueSentinelTokenListenerQueue)
import loader.models.shared as shared
from loader.models.loader.args import parser
from loader.models.loader import LoaderCheckPoint
from finetune.pulse_utils import pulse_train_model, stop_train_process
import shutil
import time
import datetime
import re
import os
import glob

def get_file_modify_time(filename):
    try:
        return datetime.datetime.fromtimestamp(os.stat(filename).st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print('Failed to get modification time for {}'.format(filename))
        print(e)
        return 'not available'

def get_model_update_time(model_name, lora_name):
    if 'pulse' in model_name.lower():
        update_time = get_file_modify_time(f"finetune/pulse/output/{lora_name}/adapter_model.bin")
    else:
        update_time = 'not available'
    return update_time

def on_train(model_name, lora_name, training_data_file):
    training_data_path = 'data/'+os.path.basename(training_data_file.name)
    if 'pulse' in model_name.lower():
        msg = pulse_train_model(model_name, lora_name, training_data_path)
    else:
        msg = 'please select one model!'
    return msg
nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


def upload_file(file):
    print('file',file)
    if not os.path.exists("data"):
        os.mkdir("data")
    filename = os.path.basename(file.name)
    shutil.move(file.name, "data/" + filename)
    # file_list首位插入新上传的文件
    filedir = "data/" + filename
    return filedir


def get_vs_list():
    lst_default = []
    if not os.path.exists(VS_ROOT_PATH):
        return lst_default
    lst = os.listdir(VS_ROOT_PATH)
    if not lst:
        return lst_default
    lst.sort()
    return lst_default + lst



def get_yaml_files(folder_path):
    yaml_files = glob.glob(os.path.join(folder_path, '*.yaml'))
    file_names = [os.path.splitext(os.path.basename(file))[0] for file in yaml_files]
    return file_names

yaml_files = get_yaml_files('configs')


vs_list = get_vs_list()


embedding_model_dict_list = list(embedding_model_dict.keys())

llm_model_dict_list = list(llm_model_dict.keys())


local_doc_qa = LocalDocQA()

flag_csv_logger = gr.CSVLogger()

def change_config(model_name, lora_name, knowledge_set_name):
    args_dict = {'model':model_name, 'lora':lora_name} if lora_name != '不使用' else {'model':model_name}
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    llm_model_ins = shared.loaderLLM()
    llm_model_ins.set_history_len(LLM_HISTORY_LEN)
    local_doc_qa.init_cfg(llm_model=llm_model_ins)
    if local_doc_qa.embeddings is None:
        local_doc_qa.init_embedding()
    return [[None, '模型加载完成']]

def read_config(ass_name_en):
    config_file_path = f"configs/{ass_name_en}.yaml"
    with open(config_file_path, 'r', encoding='utf-8') as file:
        yaml = ruamel.yaml.YAML()
        config = yaml.load(file)
    ass_name = config['ass_name']
    llm_model = config['llm_model']
    embedding_model = config['embedding_model']
    llm_history_len = config['llm_history_len']
    lora_name = config['lora_name']
    top_k = config['top_k']
    score_threshold = config['score_threshold']
    chunk_content = config['chunk_content']
    chunk_sizes = config['chunk_sizes']
    show_reference = config['show_reference']
    knowledge_set_name = config['knowledge_set_name']
    prompt_template = config['prompt_template']

    return ass_name_en,ass_name, llm_model, embedding_model, lora_name, llm_history_len, knowledge_set_name, top_k, score_threshold, chunk_content, chunk_sizes, show_reference,prompt_template

def remove_html_tags(text):
    clean_text = re.sub('<.*?>', '', text)
    return clean_text

def get_chat_answer(query, ass_id, history, score_threshold=VECTOR_SEARCH_SCORE_THRESHOLD,
               vector_search_top_k=VECTOR_SEARCH_TOP_K, chunk_content: bool = True,
               chunk_size=CHUNK_SIZE, streaming: bool = STREAMING):
    import copy
    print('===query===',query)
    ass_name_en,ass_name, llm_model, embedding_model, lora_name, llm_history_len, knowledge_set_name, top_k, score_threshold, chunk_content, chunk_sizes,show_reference,prompt_template = read_config(ass_id)
    history[-1][-1] = remove_html_tags(history[-1][-1])
    history = history[-llm_history_len:] if history is not None and len(history) > llm_history_len else history
    print('===history===',history)
    local_doc_qa.top_k = top_k
    local_doc_qa.score_threshold = score_threshold
    local_doc_qa.chunk_content = chunk_content
    local_doc_qa.chunk_size = chunk_size
    if local_doc_qa.llm is None:
        args_dict = {'model':llm_model, 'lora':lora_name} if lora_name != '不使用' else {'model':llm_model}
        shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
        llm_model_ins = shared.loaderLLM()
        llm_model_ins.set_history_len(llm_history_len)
        local_doc_qa.init_cfg(llm_model=llm_model_ins)

    if knowledge_set_name !='不使用知识库':
        vs_path = os.path.join(VS_ROOT_PATH, knowledge_set_name)
        print('vs_path',vs_path)
        if local_doc_qa.embeddings is None:
            local_doc_qa.init_embedding(model_name=embedding_model)
        for resp, history in local_doc_qa.get_knowledge_based_answer(model_name=llm_model,
                query=query, vs_path=vs_path, prompt_template=prompt_template,chat_history=history, streaming=streaming):
            if len(resp["source_documents"])>0:
                source = ""
                source += "".join(
                    [f"""出处 [{i + 1}] {os.path.split(doc.metadata["source"])[-1]}\n"""
                    f"""{doc.page_content}\n"""
                    for i, doc in
                    enumerate(resp["source_documents"])])
            else:
                source = "暂无"
            
            yield history, "", source
    else:
        print('纯聊天模式')
        for answer_result in local_doc_qa.llm.generatorAnswer(prompt=query, history=history,
                                                              streaming=streaming):

            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][-1] = resp 
            yield history, "", ""

def get_knowledge_search(query, vs_path, history, score_threshold=VECTOR_SEARCH_SCORE_THRESHOLD,
               vector_search_top_k=VECTOR_SEARCH_TOP_K, chunk_content: bool = True,
               chunk_size=CHUNK_SIZE, streaming: bool = STREAMING):
    if local_doc_qa.embeddings is None:
            local_doc_qa.init_embedding()
    if os.path.exists(vs_path):
        resp, prompt = local_doc_qa.get_knowledge_based_conent_test(query=query, vs_path=vs_path,
                                                                    score_threshold=score_threshold,
                                                                    vector_search_top_k=vector_search_top_k,
                                                                    chunk_content=chunk_content,
                                                                    chunk_size=chunk_size)
        if not resp["source_documents"]:
            yield history + [[query,
                                "根据您的设定，没有匹配到任何内容，请确认您设置的知识相关度 Score 阈值是否过小或其他参数是否正确。"]], ""
        else:
            source = "\n".join(
                [
                    f"""<details open> <summary>【知识相关度 Score】：{doc.metadata["score"]} - 【出处{i + 1}】：  {os.path.split(doc.metadata["source"])[-1]} </summary>\n"""
                    f"""{doc.page_content}\n"""
                    f"""</details>"""
                    for i, doc in
                    enumerate(resp["source_documents"])])
            history.append([query, "以下内容为知识库中满足设置条件的匹配结果：\n\n" + source])
            yield history, ""
    else:
        yield history + [[query,
                            "请选择知识库后进行测试，当前未选择知识库。"]], ""

def get_answer(query, vs_path, history, mode, score_threshold=VECTOR_SEARCH_SCORE_THRESHOLD,
               vector_search_top_k=VECTOR_SEARCH_TOP_K, chunk_content: bool = True,
               chunk_size=CHUNK_SIZE, streaming: bool = STREAMING):
    if mode == "知识库问答" and vs_path is not None and os.path.exists(vs_path):
        if local_doc_qa.embeddings is None:
            local_doc_qa.init_embedding()
        for resp, history in local_doc_qa.get_knowledge_based_answer(
                query=query, vs_path=vs_path, chat_history=history, streaming=streaming):
            source = "\n\n"
            source += "".join(
                [f"""<details> <summary>出处 [{i + 1}] {os.path.split(doc.metadata["source"])[-1]}</summary>\n"""
                 f"""{doc.page_content}\n"""
                 f"""</details>"""
                 for i, doc in
                 enumerate(resp["source_documents"])])
            history[-1][-1] += source
            yield history, ""
    elif mode == "知识库配置":
        if local_doc_qa.embeddings is None:
            local_doc_qa.init_embedding()
        if os.path.exists(vs_path):
            resp, prompt = local_doc_qa.get_knowledge_based_conent_test(query=query, vs_path=vs_path,
                                                                        score_threshold=score_threshold,
                                                                        vector_search_top_k=vector_search_top_k,
                                                                        chunk_content=chunk_content,
                                                                        chunk_size=chunk_size)
            if not resp["source_documents"]:
                yield history + [[query,
                                  "根据您的设定，没有匹配到任何内容，请确认您设置的知识相关度 Score 阈值是否过小或其他参数是否正确。"]], ""
            else:
                source = "\n".join(
                    [
                        f"""<details open> <summary>【知识相关度 Score】：{doc.metadata["score"]} - 【出处{i + 1}】：  {os.path.split(doc.metadata["source"])[-1]} </summary>\n"""
                        f"""{doc.page_content}\n"""
                        f"""</details>"""
                        for i, doc in
                        enumerate(resp["source_documents"])])
                history.append([query, "以下内容为知识库中满足设置条件的匹配结果：\n\n" + source])
                yield history, ""
        else:
            yield history + [[query,
                              "请选择知识库后进行测试，当前未选择知识库。"]], ""
    else:
        for answer_result in local_doc_qa.llm.generatorAnswer(prompt=query, history=history,
                                                              streaming=streaming):

            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][-1] = resp + (
                "\n\n当前知识库为空，如需基于知识库进行问答，请先加载知识库后，再进行提问。" if mode == "知识库问答" else "")
            yield history, ""
    # logger.info(f"flagging: username={FLAG_USER_NAME},query={query},vs_path={vs_path},mode={mode},history={history}")
    # flag_csv_logger.flag([query, vs_path, history, mode], username=FLAG_USER_NAME)

def change_assistant_input(ass_id):

    ass_name_en,ass_name, llm_model, embedding_model, lora_name, llm_history_len, knowledge_set_name, top_k, score_threshold, chunk_content, chunk_sizes,show_reference,prompt_template = read_config(ass_id)

    init_hello = f"你好，我是{ass_name}"

    if show_reference:
        return [[None,init_hello]], gr.update(visible=True)
    else:
        return [[None,init_hello]], gr.update(visible=False)



import ruamel.yaml
def set_config(ass_name_en,ass_name, llm_model, embedding_model, lora_name, llm_history_len, knowledge_set_name, top_k, score_threshold, chunk_content, chunk_sizes,show_reference, prompt_template, ass_list):
    config = {
        'ass_name':ass_name,
        'llm_model': llm_model,
        'embedding_model': embedding_model,
        'lora_name': lora_name,
        'llm_history_len': llm_history_len,
        'knowledge_set_name':knowledge_set_name,
        'top_k': top_k,
        'score_threshold':score_threshold,
        'chunk_content':chunk_content,
        'chunk_sizes':chunk_sizes,
        'show_reference':show_reference,
        'prompt_template':prompt_template
    }
    yaml = ruamel.yaml.YAML()
    with open(f'configs/{ass_name_en}.yaml', 'w', encoding="utf-8") as file:
        yaml.dump(config, file)

    return gr.update(visible=True),f'configs/{ass_name_en}.yaml 保存成功!',gr.update(visible=True, choices=ass_list, value=ass_list[0])


def get_vector_store(vs_id, files, sentence_size, history, one_conent, one_content_segmentation):
    if local_doc_qa.embeddings is None:
        local_doc_qa.init_embedding()

    vs_path = os.path.join(VS_ROOT_PATH, vs_id)
    filelist = []
    if not os.path.exists(os.path.join(UPLOAD_ROOT_PATH, vs_id)):
        os.makedirs(os.path.join(UPLOAD_ROOT_PATH, vs_id))
    if isinstance(files, list):
        for file in files:
            filename = os.path.split(file.name)[-1]
            shutil.move(file.name, os.path.join(UPLOAD_ROOT_PATH, vs_id, filename))
            filelist.append(os.path.join(UPLOAD_ROOT_PATH, vs_id, filename))
            print('\n======filelist, vs_path=========\n',filelist, vs_path)
        vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(filelist, vs_path, sentence_size)
    else:
        vs_path, loaded_files = local_doc_qa.one_knowledge_add(vs_path, files, one_conent, one_content_segmentation,
                                                                sentence_size)
    if len(loaded_files):
        file_status = f"已添加 {'、'.join([os.path.split(i)[-1] for i in loaded_files if i])} 内容至知识库，并已加载知识库，请开始提问"
    else:
        file_status = "文件未成功加载，请重新上传文件"
    # if local_doc_qa.llm and local_doc_qa.embeddings:
       
    # else:
    #     file_status = "模型未完成加载，请先在加载模型后再导入文件"
    #     vs_path = None
    logger.info(file_status)
    return vs_path, None, history + [[None, file_status]]


def change_vs_name_input(vs_id, history):
    if vs_id == "新建知识库":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), None, history
    else:
        file_status = f"已加载知识库{vs_id}，请开始提问"
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), os.path.join(VS_ROOT_PATH,
                                                                                                         vs_id), history + [
                   [None, file_status]]


knowledge_base_test_mode_info = ("【注意】\n\n"
                                 "1.您已进入知识库测试模式，仅用于测试知识库相关参数配置\n\n"
                                 "2.知识相关度 Score，建议设置为 500~800，具体设置情况请结合实际使用调整。\n\n"
                                 "3.目前支持的处理格式包含非图片格式的pdf、word、txt、md、excel、json、jsonl格式\n\n"
                                 "其中excel格式可包含多个数据表，每个数据表里必须包含两列：问题|回答\n\n"
                                 """其中json、jsonl格式处理成多行，每行一个dict类似这样：{"docs": ["氨力农注射液 药物分类\n化学药品", "氨力农注射液 药物剂量\n10毫升:50毫克"]}""")


def change_mode(mode, history):
    if mode == "知识库问答":
        return gr.update(visible=True), gr.update(visible=False), history
        # + [[None, "【注意】：您已进入知识库问答模式，您输入的任何查询都将进行知识库查询，然后会自动整理知识库关联内容进入模型查询！！！"]]
    elif mode == "知识库配置":
        return gr.update(visible=True), gr.update(visible=True), [[None,
                                                                   knowledge_base_test_mode_info]]
    else:
        return gr.update(visible=False), gr.update(visible=False), history


def change_chunk_content(mode, label_conent, history):
    conent = ""
    if "chunk_content" in label_conent:
        conent = "搜索结果上下文关联"
    elif "one_content_segmentation" in label_conent:  # 这里没用上，可以先留着
        conent = "内容分段入库"

    if mode:
        return gr.update(visible=True), history + [[None, f"【已开启{conent}】"]]
    else:
        return gr.update(visible=False), history + [[None, f"【已关闭{conent}】"]]


def add_vs_name(vs_name, vs_list, chatbot):
    if not os.path.exists(VS_ROOT_PATH):
        os.makedirs(VS_ROOT_PATH)
    if vs_name in vs_list:
        vs_status = "与已有知识库名称冲突，请重新选择其他名称后提交"
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True), vs_list, gr.update(visible=True), gr.update(visible=True), gr.update(
            visible=False), chatbot
    else:
        vs_status = f"""已新增知识库"{vs_name}",将在上传文件并载入成功后进行存储。请在开始对话前，先完成文件上传。 """
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True, choices=[vs_name] + vs_list, value=vs_name), [vs_name] + vs_list, gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=True), chatbot

def change_lora_name_input(model_name,lora_name_en):
    if lora_name_en == "新建Lora":
        return gr.update(visible=True), gr.update(visible=True)
    else:
        file_status = f"已加载{lora_name_en}"
        model_update_time = get_model_update_time(model_name, lora_name_en)
        print('lora_name_en',lora_name_en)
        return gr.update(visible=False), gr.update(visible=False), model_update_time


def add_lora(lora_name_en,lora_list):
    if lora_name_en in lora_list:
        print('名称冲突，不新建')
        return gr.update(visible=True,value=lora_name_en), gr.update(visible=False), gr.update(visible=False), lora_list
    else:
        return gr.update(visible=True, choices=[lora_name_en] + lora_list, value=lora_name_en), gr.update(visible=False), gr.update(visible=False),[lora_name_en] + lora_list

def change_assistant_name_input(ass_id):
    if ass_id == "新建小助手":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), '医疗小助手', LLM_MODEL,EMBEDDING_MODEL,'不使用',LLM_HISTORY_LEN,cur_vs_list.value[0] if len(cur_vs_list.value) > 1 else '不使用知识库',VECTOR_SEARCH_TOP_K,500,True,250,True,''

    else:
        file_status = f"已加载{ass_id}"
        print('file_status',file_status)
        ass_name_en,ass_name, llm_model, embedding_model, lora_name, llm_history_len, knowledge_set_name, top_k, score_threshold, chunk_content, chunk_sizes,show_reference,prompt_template = read_config(ass_id)
        
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), ass_name, llm_model, embedding_model, lora_name, llm_history_len, knowledge_set_name, top_k, score_threshold, chunk_content, chunk_sizes,show_reference,prompt_template


def add_ass_config(ass_id,ass_list):
    if ass_id in ass_list:
        print('名称冲突，不新建')
        return ass_id, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), '医疗小助手', LLM_MODEL,EMBEDDING_MODEL,'不使用',LLM_HISTORY_LEN,cur_vs_list.value[0] if len(cur_vs_list.value) > 1 else '不使用知识库',VECTOR_SEARCH_TOP_K,500,True,250,True,'',ass_list,gr.update(visible=True)
    else:
        return ass_id, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), '医疗小助手', LLM_MODEL,EMBEDDING_MODEL,'不使用',LLM_HISTORY_LEN,cur_vs_list.value[0] if len(cur_vs_list.value) > 1 else '不使用知识库',VECTOR_SEARCH_TOP_K,500,True,250,True,'',ass_list+[ass_id],gr.update(visible=True, choices=ass_list+[ass_id], value=ass_id)

def find_folders(directory):
    folders = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folders.append(item)
    return folders

def change_model_name_input(model_name):
    if 'pulse' in model_name.lower():
        model_name = 'pulse'
    else:
        model_name = ''
    model_dir = os.path.join(f"finetune", model_name,'output')
    lora_list = find_folders(model_dir)
    return lora_list,gr.update(visible=True, choices=lora_list+["新建Lora"], value=lora_list[0] if len(lora_list)>0 else "新建Lora")

def change_model_name_select(model_name):
    if 'pulse' in model_name.lower():
        model_name = 'pulse'
    else:
        model_name = ''
    model_dir = os.path.join(f"finetune", model_name,'output')
    lora_list = find_folders(model_dir)
    return lora_list,gr.update(visible=True, choices=lora_list+["不使用"], value=lora_list[0] if len(lora_list)>0 else "不使用")

block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""

webui_title = """
# 💁医疗知识聊天机器人💁
"""
default_vs = vs_list[0] if len(vs_list) > 1 else "为空"
init_message = f"""请先在右侧选择小助手，再开始对话测试
"""

# 初始化消息
args = None
args = parser.parse_args()


model_status = '请手动加载模型'

default_theme_args = dict(
    font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
)

def get_lora_init_list(model_name):
    if 'pulse' in model_name.lower():
        model_name = 'pulse'
    else:
        model_name = ''
    model_dir = os.path.join(f"finetune", model_name,'output')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    lora_list = find_folders(model_dir)
    return lora_list

lora_init_list = get_lora_init_list(llm_model_dict_list[0])

with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args)) as demo:
    vs_path, file_status, model_status, set_vs_list , cur_vs_list, set_lora_list, set_ass_list = gr.State(
        os.path.join(VS_ROOT_PATH, vs_list[0]) if len(vs_list) > 1 else ""), gr.State(""), gr.State(
        model_status), gr.State(vs_list+['新建知识库']), gr.State(vs_list+['不使用知识库']), gr.State(lora_init_list), gr.State(yaml_files)

    gr.Markdown(webui_title)
    with gr.Tab("对话测试"):
        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot([[None, init_message]],
                                     elem_id="chat-box",
                                     show_label=False).style(height=750)
                query = gr.Textbox(show_label=False,
                                   placeholder="请输入提问内容，按回车进行提交").style(container=False)
            with gr.Column(scale=5):
                choose_ass = gr.Dropdown(yaml_files,
                                        label="选择要使用的小助手",
                                        value= yaml_files if len(yaml_files) > 1 else '暂无可使用的',
                                        interactive=True)
                reference = gr.Textbox(type="text", label='参考资料',visible=True)
            choose_ass.change(fn=change_assistant_input,
                                     inputs=[choose_ass],
                                     outputs=[chatbot,reference])
            query.submit(get_chat_answer,
                            [query, choose_ass, chatbot],
                            [chatbot, query, reference])

    with gr.Tab("知识库测试"):
        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot([[None, knowledge_base_test_mode_info]],
                                     elem_id="chat-box",
                                     show_label=False).style(height=750)
                query = gr.Textbox(show_label=False,
                                   placeholder="请输入提问内容，按回车进行提交").style(container=False)
            with gr.Column(scale=5):
                knowledge_set = gr.Accordion("知识库设定", visible=True)
                vs_setting = gr.Accordion("配置知识库", visible=True)
                with knowledge_set:
                    score_threshold = gr.Number(value=VECTOR_SEARCH_SCORE_THRESHOLD,
                                                label="知识相关度 Score 阈值，分值越低匹配度越高",
                                                precision=0,
                                                interactive=True)
                    vector_search_top_k = gr.Number(value=VECTOR_SEARCH_TOP_K, precision=0,
                                                    label="获取知识库内容条数", interactive=True)
                    chunk_content = gr.Checkbox(value=False,
                                               label="是否启用上下文关联",
                                               interactive=True)
                    chunk_sizes = gr.Number(value=CHUNK_SIZE, precision=0,
                                            label="匹配单段内容的连接上下文后最大长度",
                                            interactive=True, visible=True)
                    chunk_content.change(fn=change_chunk_content,
                                        inputs=[chunk_content, gr.Textbox(value="chunk_content", visible=False), chatbot],
                                        outputs=[chunk_sizes, chatbot])
                with vs_setting:
                    select_vs = gr.Dropdown(set_vs_list.value,
                                            label="请选择要加载的知识库",
                                            interactive=True,
                                            value=set_vs_list.value[0] if len(set_vs_list.value) > 0 else None)
                    vs_name = gr.Textbox(label="请输入新建知识库名称，当前知识库命名暂不支持中文",
                                         lines=1,
                                         interactive=True,
                                         visible=True)
                    vs_add = gr.Button(value="添加至知识库选项", visible=True)
                    file2vs = gr.Column(visible=False)
                    with file2vs:
                        # load_vs = gr.Button("加载知识库")
                        gr.Markdown("向知识库中添加单条内容或文件")
                        sentence_size = gr.Number(value=SENTENCE_SIZE, precision=0,
                                                  label="文本入库分句长度限制",
                                                  interactive=True, visible=True)
                        with gr.Tab("上传文件"):
                            files = gr.File(label="添加文件",
                                            file_types=['.txt', '.md', '.docx', '.pdf', '.jsonl'],
                                            file_count="multiple",
                                            show_label=False
                                            )
                            load_file_button = gr.Button("上传文件并加载知识库")
                        with gr.Tab("上传文件夹"):
                            folder_files = gr.File(label="添加文件",
                                                   # file_types=['.txt', '.md', '.docx', '.pdf'],
                                                   file_count="directory",
                                                   show_label=False)
                            load_folder_button = gr.Button("上传文件夹并加载知识库")
                        with gr.Tab("添加单条内容"):
                            one_title = gr.Textbox(label="标题", placeholder="请输入要添加单条段落的标题", lines=1)
                            one_conent = gr.Textbox(label="内容", placeholder="请输入要添加单条段落的内容", lines=5)
                            one_content_segmentation = gr.Checkbox(value=True, label="禁止内容分句入库",
                                                                   interactive=True)
                            load_conent_button = gr.Button("添加内容并加载知识库")
                    # 将上传的文件保存到content文件夹下,并更新下拉框
                    vs_add.click(fn=add_vs_name,
                                 inputs=[vs_name, set_vs_list, chatbot],
                                 outputs=[select_vs, set_vs_list, vs_name, vs_add, file2vs, chatbot])
                    select_vs.change(fn=change_vs_name_input,
                                     inputs=[select_vs, chatbot],
                                     outputs=[vs_name, vs_add, file2vs, vs_path, chatbot])
                    load_file_button.click(get_vector_store,
                                           show_progress=True,
                                           inputs=[select_vs, files, sentence_size, chatbot, vs_add, vs_add],
                                           outputs=[vs_path, files, chatbot], )
                    load_folder_button.click(get_vector_store,
                                             show_progress=True,
                                             inputs=[select_vs, folder_files, sentence_size, chatbot, vs_add,
                                                     vs_add],
                                             outputs=[vs_path, folder_files, chatbot], )
                    load_conent_button.click(get_vector_store,
                                             show_progress=True,
                                             inputs=[select_vs, one_title, sentence_size, chatbot,
                                                     one_conent, one_content_segmentation],
                                             outputs=[vs_path, files, chatbot], )
                    # flag_csv_logger.setup([query, vs_path, chatbot, mode], "flagged")
                    query.submit(get_knowledge_search,
                                 [query, vs_path, chatbot, score_threshold, vector_search_top_k, chunk_content,
                                  chunk_sizes],
                                 [chatbot, query])
    with gr.Tab("lora微调"):
        with gr.Row():
            with gr.Column():
                    model_name = gr.Radio(llm_model_dict_list, #'Bert',
                                            label="选择模型",
                                            value= llm_model_dict_list[0] if len(llm_model_dict_list)>0 else '暂无可选模型',
                                            interactive=True)
            with gr.Column():
                select_lora = gr.Dropdown(set_lora_list.value+['新建Lora'],
                                        label= "选择或者新建一个Lora",
                                        value= set_lora_list.value[0] if len(set_lora_list.value) > 0 else '新建Lora',
                                        interactive=True)
                lora_name_en = gr.Textbox(label="请输入Lora英文名称，中间不能有空格，小写字母，单词间可用下划线分开",
                                            lines=1,
                                            interactive=True,
                                            visible=False)
                lora_add = gr.Button(value="确认添加Lora", visible=False)
        with gr.Row():
            lastest_model = gr.outputs.Textbox(type="text", label='模型更新时间（请切换模型或Lora刷新显示）')
        gr.Markdown("## lora微调，目前只支持excel格式，要求语料格式为问题|回答两列，或者系统指示|问题|回答三列")
        train_data_file = gr.File(label="上传对话语料文件", file_types=['.xlsx'])
        train_button = gr.Button("开始训练", label="训练")
        kill_train_button = gr.Button("停止所有训练进程", label="训练")
        train_res = gr.outputs.Textbox(type="text", label='')
        train_data_file.upload(upload_file,
                inputs=train_data_file)
        train_button.click(on_train, inputs=[model_name, select_lora, train_data_file],outputs=[train_res])
        model_name.change(fn=change_model_name_input,
                                     inputs=[model_name],
                                     outputs=[set_lora_list,select_lora])
        select_lora.change(fn=change_lora_name_input,
                                     inputs=[model_name,select_lora],
                                     outputs=[lora_name_en, lora_add,lastest_model])
        lora_add.click(fn=add_lora,
                                 inputs=[lora_name_en,set_lora_list],
                                 outputs=[select_lora, lora_name_en, lora_add,set_lora_list])

    with gr.Tab("医疗小助手配置"):
        with gr.Column():
            select_ass = gr.Dropdown(set_ass_list.value+['新建小助手'],
                                        label="选择或者新建一个医疗小助手",
                                        value= set_ass_list.value[0] if len(set_ass_list.value) > 0 else '新建小助手',
                                        interactive=True)
            ass_name_en = gr.Textbox(label="请输入小助手英文名称，中间不能有空格，小写字母，单词间可用下划线分开",
                                         lines=1,
                                         interactive=True,
                                         visible=False)
            ass_add = gr.Button(value="确认添加小助手", visible=False)
        ass_config = gr.Column(visible=False)
        with ass_config:
            ass_name = gr.Textbox(label="请给机器人取个名字，随便取，中英文均可，可以有空格",
                                            value='医疗小助手',
                                            lines=1,
                                            interactive=True,
                                            visible=True)
            llm_model = gr.Radio(llm_model_dict_list,
                                label="LLM 模型",
                                value=llm_model_dict_list[0] if len(llm_model_dict_list)>0 else '暂无可选模型',
                                interactive=True)
            embedding_model = gr.Radio(embedding_model_dict_list,
                                    label="Embedding 模型",
                                    value=EMBEDDING_MODEL,
                                    interactive=True)
            lora_name = gr.Dropdown(set_lora_list.value+['不使用'],
                                value='不使用',
                                label="选择使用的Lora",
                                interactive=True)
            llm_history_len = gr.Slider(0, 10,
                                        value=LLM_HISTORY_LEN,
                                        step=1,
                                        label="LLM 对话轮数",
                                        interactive=True)
            knowledge_set_name = gr.Dropdown(cur_vs_list.value,
                                            label="选择知识库",
                                            value= cur_vs_list.value[0] if len(cur_vs_list.value) > 1 else '不使用知识库',
                                            interactive=True)
            top_k = gr.Slider(1, 20, value=VECTOR_SEARCH_TOP_K, step=1,
                            label="向量匹配 top k", interactive=True)
            score_threshold = gr.Number(value=700,label="知识相关度 Score 阈值,分值越低匹配度越高,数值范围约为0-1100,一般在700左右",
                                                    precision=0,
                                                    interactive=True)
            chunk_content = gr.Checkbox(value=True,
                                        label="是否启用上下文关联",
                                        interactive=True)
            chunk_sizes = gr.Number(value=250, precision=0,
                                    label="匹配单段内容的连接上下文后最大长度",
                                    interactive=True, visible=True)
            show_reference = gr.Checkbox(value=True,
                                        label="是否显示参考文献窗口",
                                        interactive=True)
            prompt_note = """prompt_template ，{context} 代表搜出来的文档，{chat_history}代表历史聊天记录，{question}代表最后一个问题，请在prompt里加上这些关键词。注意不使用知识库的情况下不生效。参考例子：假设你是用药助手，请根据文档来回复，如果文档内容为空或者None，则忽略，文档:{context}\n{chat_history}</s>User: {question}</s>"""
            gr.Markdown(prompt_note)
            prompt_template = gr.Textbox(label="内置prompt模板",
                                            value="假设你是用药助手，请根据文档来回复，如果文档内容为空或者None，则忽略，文档:{context}\n{chat_history}</s>User: {question}</s>",
                                            lines=8,
                                            interactive=True,
                                            visible=True)
            save_config_button = gr.Button("保存助手配置")
            save_res = gr.Textbox(type="text", label='', visible=False)
        save_config_button.click(set_config, show_progress=True,
                                inputs=[select_ass,ass_name, llm_model, embedding_model, lora_name, llm_history_len, knowledge_set_name, top_k, score_threshold, chunk_content, chunk_sizes, show_reference, prompt_template,set_ass_list], outputs=[save_res,save_res,choose_ass])
                                        
        llm_model.change(fn=change_model_name_select,
                                     inputs=[llm_model],
                                     outputs=[set_lora_list,lora_name])
        select_ass.change(fn=change_assistant_name_input,
                                     inputs=[select_ass],
                                     outputs=[ass_name_en, ass_add, ass_config, save_res, ass_name, llm_model, embedding_model, lora_name, llm_history_len, knowledge_set_name, top_k, score_threshold, chunk_content, chunk_sizes, show_reference, prompt_template])
        ass_add.click(fn=add_ass_config,
                                 inputs=[ass_name_en,set_ass_list],
                                 outputs=[select_ass, ass_name_en, ass_add, ass_config, save_res, ass_name, llm_model, embedding_model, lora_name, llm_history_len, knowledge_set_name, top_k, score_threshold, chunk_content, chunk_sizes, show_reference, prompt_template,set_ass_list,select_ass])
(demo
 .queue(concurrency_count=3)
 .launch(server_name='0.0.0.0',
         server_port=3355,
         show_api=False,
         share=True,
         debug= True,
         inbrowser=True))