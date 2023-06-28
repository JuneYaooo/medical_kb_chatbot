import logging
from typing import List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.helpers import detect_file_encodings
import json
import pandas as pd

logger = logging.getLogger(__name__)

def process_json(json_str):
    docs=[]
    for doc in json_str['docs']:
        new_doc = Document(
            page_content=doc, metadata={"source": doc.split(' ')[0]}
        )
        docs.append(new_doc)
        
    return docs

class ExcelLoader(BaseLoader):
    """Load text files.


    Args:
        file_path: Path to the file to load.

        encoding: File encoding to use. If `None`, the file will be loaded
        with the default system encoding.

        autodetect_encoding: Whether to try to autodetect the file encoding
            if the specified encoding fails.
    """

    def __init__(
        self,
        file_path: str,
        encoding: Optional[str] = None,
        autodetect_encoding: bool = False,
    ):
        """Initialize with file path."""
        self.file_path = file_path
        self.encoding = encoding
        self.autodetect_encoding = autodetect_encoding

    def load(self) -> List[Document]:
        """Load from file path."""
        pass

    def load_excel(self) -> List[Document]:
        """Load from file path."""
        documents = []
        try:
            df = pd.read_excel(self.file_path, sheet_name=None)
            # 获取所有sheet的名字
            sheet_names = df.keys()
            # 遍历每个sheet
            for sheet_name in sheet_names:
                # 获取当前sheet的数据
                sheet_data = df[sheet_name]
                # 遍历每行数据
                for _, row in sheet_data.iterrows():
                    try:
                        question = str(row["问题"])
                        answer = str(row["回答"])
                        doc = "【参考问题】"+question+"【参考回答】"+answer
                        new_doc = Document(
                            page_content=doc, metadata={"source": sheet_name}
                        )
                        documents.append(new_doc)
                    except Exception as e:
                        print('文件表读取发生错误！',sheet_name,e)
        except Exception as e:
            print('e2',e)
            raise RuntimeError(f"Error loading {self.file_path}") from e
        return documents