from easyEval.Model.api import LLM_API
from easyEval.Model.llm import LLM_Model
 
API_MODEL = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]

str2models = { "api": LLM_API, "local": LLM_Model}



__all__ = ["str2datasets"]
