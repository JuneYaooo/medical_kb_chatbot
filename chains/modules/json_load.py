import logging
from typing import List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.helpers import detect_file_encodings
import json
logger = logging.getLogger(__name__)

def process_json(json_str):
    docs=[]
    for doc in json_str['docs']:
        new_doc = Document(
            page_content=doc, metadata={"source": doc.split(' ')[0]}
        )
        docs.append(new_doc)
        
    return docs

class JsonLoader(BaseLoader):
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

    def load_json(self) -> List[Document]:
        """Load from file path."""
        documents = []
        try:
            with open(self.file_path, encoding=self.encoding) as f:
                for line in f:
                    json_object = json.loads(line)
                    docs = process_json(json_object)
                    documents+= docs
        except UnicodeDecodeError as e:
            print('e1',e)
            if self.autodetect_encoding:
                detected_encodings = detect_file_encodings(self.file_path)
                for encoding in detected_encodings:
                    logger.debug("Trying encoding: ", encoding.encoding)
                    try:
                        with open(self.file_path, 'r', encoding=encoding.encoding) as f:
                            for line in f:
                                json_object = json.loads(line)
                                docs = process_json(json_object)
                                documents+= docs
                        break
                    except UnicodeDecodeError:
                        continue
            else:
                raise RuntimeError(f"Error loading {self.file_path}") from e
        except Exception as e:
            print('e2',e)
            raise RuntimeError(f"Error loading {self.file_path}") from e
        return documents