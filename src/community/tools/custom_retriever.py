from typing import Any, Dict, List
from dotenv import load_dotenv


from community.tools import BaseTool

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pymongo import MongoClient
import os

# cohere-toolkit\.env
load_dotenv()

# load_dotenv()

ATLAS_CONNECTION_STRING = os.getenv("MONGO_DB_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class CustomRetriever(BaseTool):
    NAME = "mongodb_atlas"
    
    def __init__(self):

        self.client = MongoClient(ATLAS_CONNECTION_STRING)

    @classmethod
    # If your tool requires any environment variables such as API keys,
    # you will need to assert that they're not None here
    def is_available(cls) -> bool:
        return True

    #  Your tool needs to implement this call() method
    async def call(self, parameters: str, **kwargs: Any) -> List[Dict[str, Any]]:
        
        db_name = "tech_innovators_db"
        collection_name = "tech_innovators_collection"
        atlas_collection = self.client[db_name][collection_name]
        vector_search_index = "vector_index_erp"

        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
        vectorstore = MongoDBAtlasVectorSearch(
            embedding = embeddings,
            collection = atlas_collection,
            index_name = vector_search_index)
        
        response = []
        query = parameters.get("query", "")
        documents = vectorstore.similarity_search(query)
        my_result_lst = []
        my_result = {}
        
        return [
            (
                {
                "text": doc.page_content,
                "pageid": doc.metadata.get("pageid", None),
                "department": doc.metadata.get("department", None),
                }
            )
            for doc in documents
        ]


