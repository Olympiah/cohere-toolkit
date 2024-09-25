from typing import Any, Dict, List

from backend.chat.collate import to_dict
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from backend.schemas.cohere_chat import CohereChatRequest
from backend.schemas.context import Context
from community.model_deployments import BaseDeployment
from backend.services.logger.utils import LoggerFactory
import os
# import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

class GeminiDeployment(BaseDeployment):
    """
    Gemini deployment
    """

    DEFAULT_MODELS = [
        "gemini-1.5-flash",
    ]

    def __init__(self, **kwargs: Any):
        # self.client = genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.client = ChatGoogleGenerativeAI(api_key=os.getenv("GEMINI_API_KEY"))

    @property
    def rerank_enabled(self) -> bool:
        return False

    @classmethod
    def list_models(cls) -> List[str]:
        if not GeminiDeployment.is_available():
            return []

        return cls.DEFAULT_MODELS

    @classmethod
    def is_available(cls) -> bool:
        return True
    
    async def invoke_chat(self, chat_request: CohereChatRequest, ctx: Context, **kwargs: Any) -> Any:
        
        logger = LoggerFactory().get_logger()
        logger.warning(event=f"[Gemini Deployment] code is in Invoke Chat now",)
        # model_id = self.DEFAULT_MODELS[0]
        model_name = chat_request.model
        temperature = chat_request.temperature
        model = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, api_key=os.getenv("GEMINI_API_KEY"))
        logger.warning(event=f"[Gemini Deployment] model successfully instantiated !!",)

        if chat_request.max_tokens is None:
            chat_request.max_tokens = 200

        response = model.invoke(
            chat_request.message,
            stream=False,
            max_tokens=chat_request.max_tokens,
            temperature=chat_request.temperature,
        )

        return {"text": response["choices"][0]["text"]}

        logger.warning(event=f"[Gemini Deployment] chat messages are successfully build {messages}",)

    async def invoke_chat_stream(
        self, chat_request: CohereChatRequest, ctx: Context, **kwargs: Any
    ) -> Any:
        """
        Built in streamming is not supported, so this function wraps the invoke_chat function to return a single response.
        """
        logger = LoggerFactory().get_logger()
        logger.warning(event=f"[Gemini Deployment] code is here on invoke chat stream",)
        logger.info(event=f"[Gemini Deployment] request is {chat_request.model_dump(exclude={'stream', 'file_ids', 'agent_id'})}")
        model_name = chat_request.model
        temperature = chat_request.temperature
        model = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, api_key=os.getenv("GEMINI_API_KEY"))
        
        if chat_request.max_tokens is None:
            chat_request.max_tokens = 200

        logger.info(event=f"[Gemini Deployment] getting the streamed response")
        stream = model.invoke(
            chat_request.message,
            stream=True,
            max_tokens=chat_request.max_tokens,
            temperature=chat_request.temperature,
        )
        logger.info(event=f"[Gemini Deployment] stream {stream}")


        #response_messages = dict(messages)
        logger.info(event=f"[Gemini Deployment] content from stream is {stream}")


        messages = {}
        for item in stream:
            logger.info(event=f"[Gemini Deployment] stream item is {item}, type is {type(item)}")
            messages[item[0]] = item[1]
            # yield(messages[-1])
        
        logger.info(event=f"[Gemini Deployment] message dict is {messages}")

        yield {
            "event_type": "stream-start",
            "generation_id": "",
            }

        yield {
            "event_type": "text-generation",
            "text": messages["content"]
            # "text": item["choices"][0]["text"],
        }

        yield {
            "event_type": "stream-end",
            "finish_reason": "COMPLETE",
            "response":  messages
        }

        logger.info(event=f"[Gemini Deployment] response is done")


        logger.info(event=f"[Gemini Deployment] response successfully created")

    async def invoke_rerank(
        self, query: str, documents: List[Dict[str, Any]], ctx: Context, **kwargs: Any
    ) -> Any:
        return None

    def _build_chat_history(
        self, chat_history: List[Dict[str, Any]], message: str
    ) -> List[Dict[str, Any]]:
        messages = []

        for message in chat_history:
            messages.append({"role": message["role"], "content": message["message"]})

        messages.append({"role": "USER", "content": message})

        return messages


if __name__ == "__main__":
    logger = LoggerFactory().get_logger()
    logger.warning(event=f"[Gemini Deployment] code is here",)
    gemini = GeminiDeployment()
    chat_request = CohereChatRequest(
        chat_history=[
            {"role": "user", "message": "Hello!"},
            {"role": "chatbot", "message": "Hi, how can I help you?"},
        ],
        message="How are you?",
    )
    response = gemini.invoke_chat(chat_request)
