
from abc import ABC

from langchain.llms.base import LLM
from typing import Optional, List
from loader.models.loader import LoaderCheckPoint
from loader.models.base import (BaseAnswer,
                         AnswerResult,
                         AnswerResultStream,
                         AnswerResultQueueSentinelTokenListenerQueue)
import re

import transformers
from transformers.generation.streamers import BaseStreamer
from threading import Thread
from queue import Queue
from typing import Callable, Iterable, List, Optional, Tuple, Union
# import torch

class MyStreamer(BaseStreamer):

    def __init__(
            self, 
            # stop_token_ids: torch.Tensor, 
            # skip_token_count: int, 
            # max_input_length: int,
            timeout: Optional[float] = None
        ):

        # 紧急停止策略
        # self.stop_token_ids = stop_token_ids
        # self.skip_token_count = skip_token_count
        # self.max_input_length = max_input_length


        self.token_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout


    def put(self, value):
        list_value = value.tolist()
        if type(list_value[0]) == int:
            self.token_queue.put(list_value, timeout=self.timeout)

    def end(self):
        self.token_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.token_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value
            
def remove_starting_symbols(string):
    pattern = r'^[，。,.！!]+'
    result = re.sub(pattern, '', string)
    return result

def extract_content(replacement_text, string):
    pattern_output = r'输出:(.*?)(?=<\/s>)'
    pattern_helper = r'<\\s>Helper: (.*?)</s>' # r'Helper:(.*?)(?=<\/s>)'

    match_output = re.findall(pattern_output, string, re.DOTALL)
    match_helper = re.findall(pattern_helper, string, re.DOTALL)

    if match_output:
        content = match_output[-1].strip()
        return content.replace('</s>', '').replace('<\s>', '')
    elif match_helper:
        content = match_helper[-1].strip()
        return content.replace('</s>', '').replace('<\s>', '')
    else:
        replaced_string = re.sub(r'Input:.*?(?=\n)', replacement_text, string, re.DOTALL)
        replaced_string = replaced_string.replace('</s>', '').replace('<\s>', '')
        return  replaced_string

import re

def remove_prefix_suffix(string):
    pattern = r'^((?:Helper:|:)\s*)(.*?)(</s>)?$'
    string = re.sub(pattern, r'\2', string, flags=re.DOTALL)
    string = remove_starting_symbols(string).replace('</s>', '').replace('<\s>', '')
    return string

class Bloomz(BaseAnswer, LLM, ABC):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    checkPoint: LoaderCheckPoint = None
    # history = []
    history_len: int = 10

    def __init__(self, checkPoint: LoaderCheckPoint = None):
        super().__init__()
        self.checkPoint = checkPoint

    @property
    def _llm_type(self) -> str:
        return "Bloomz"

    @property
    def _check_point(self) -> LoaderCheckPoint:
        return self.checkPoint

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        pass

    def _generate_answer(self, prompt: str,
                         history: List[List[str]] = [],
                         streaming: bool = False,
                         generate_with_callback: AnswerResultStream = None) -> None:
        # Create the StoppingCriteriaList with the stopping strings
        stopping_criteria_list = transformers.StoppingCriteriaList()
        # 定义模型stopping_criteria 队列，在每次响应时将 torch.LongTensor, torch.FloatTensor同步到AnswerResult
        listenerQueue = AnswerResultQueueSentinelTokenListenerQueue()
        stopping_criteria_list.append(listenerQueue)
        model_device = next(self.checkPoint.model.parameters()).device
        if streaming:
            history += [[]]
            streamer = MyStreamer() # type: ignore
            # torch.manual_seed(23333)
            inputs = self.checkPoint.tokenizer([prompt], return_tensors="pt").input_ids.to(model_device) 

            thread = Thread(target=self.checkPoint.model.generate, kwargs=dict(
                inputs=inputs,
                # attention_mask=attention_mask.cuda(),
                # gen kargs
                num_beams=1,
                do_sample=True,
                temperature=0.7,
                top_p=self.top_p,
                top_k=9,
                eos_token_id=self.checkPoint.tokenizer.eos_token_id,
                max_length=2000,
                min_length=1,
                streamer=streamer,
            ))

            thread.start()
            # stream_resp = ''
            token_list = []
            for token in streamer:
                token_list += token
                stream_resp = self.checkPoint.tokenizer.decode(token_list)
                stream_resp = remove_prefix_suffix(stream_resp)
                history[-1] = [prompt, stream_resp]  #stream_resp.lstrip(": ")
                answer_result = AnswerResult()
                answer_result.history = history
                answer_result.llm_output = {"answer": stream_resp}
                if listenerQueue.listenerQueue.__len__() > 0:
                    answer_result.listenerToken = listenerQueue.listenerQueue.pop()
                generate_with_callback(answer_result)
        else:
            inputs = self.checkPoint.tokenizer([prompt], return_tensors="pt").input_ids.to(model_device) 
            re_token_ids = self.checkPoint.model.generate(
                inputs=inputs, 
                # gen kargs
                num_beams=1,
                do_sample=True,
                temperature=0.7,
                top_p=self.top_p,
                top_k=9,
                eos_token_id=self.checkPoint.tokenizer.eos_token_id,
                max_length=2000 #512,
            )
            response = self.checkPoint.tokenizer.decode(re_token_ids[0])
            response = extract_content(prompt, response)
            self.checkPoint.clear_torch_cache()
            history += [[prompt, response]]
            answer_result = AnswerResult()
            answer_result.history = history
            answer_result.llm_output = {"answer": response}
            if listenerQueue.listenerQueue.__len__() > 0:
                answer_result.listenerToken = listenerQueue.listenerQueue.pop()

            generate_with_callback(answer_result)