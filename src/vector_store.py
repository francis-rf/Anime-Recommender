from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader

from dotenv import load_dotenv
load_dotenv(override=True)


class VectorStoreBuilder:
    def __init__(self,csv_path:str,persist_dir:str='chroma_db'):
        self.csv_path = csv_path
        self.persist_dir = persist_dir
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
        self.embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device':'cpu'})
        
    def build_and_save_vectorstore(self):
        loader = CSVLoader(file_path=self.csv_path,encoding='utf-8',metadata_columns=[])
        data = loader.load()
        chunks = self.text_splitter.split_documents(data)
        vectorstore = Chroma.from_documents(chunks,persist_directory=self.persist_dir,embedding=self.embeddings)
        vectorstore.persist()

    def load_vectorstore(self):
        return Chroma(persist_directory=self.persist_dir,embedding_function=self.embeddings)    