import requests
import copy
from typing import Any, List, Iterable, Optional, Dict, Callable
from demeterchain.utils import MessageTemplate

class ChatCompletionsLLM(object):
    """
    Interface for calling llm.
    
    Examples:

        .. code-block:: python

            from demeterchain.llms import ChatCompletionsLLM
            from demeterchain.utils import MessageTemplate

            token = "your_token" # Please fill in your token
            host = "https://api.openai.com/v1"
            model = "gpt-4-turbo"

            template = MessageTemplate(
                input_variables = ["query"],
                template=[
                    {"role": "system", "content": "請幫我解答以下問題，請用中文回答。",},
                    {"role": "user", "content": "{query}"}
                ]
            )
            llm = ChatCompletionsLLM(
                host = host,
                model = model,
                token = token,
                template= template,
                generation_config={
                    "max_tokens": 20,
                    "temperature": 0.9,
                }
            )
            response = llm({"query": "蘋果是甚麼顏色?"})
        >>> print(response)
        蘋果有多種顏色，包括紅色、
    
    """
    def __init__(
        self,
        template: MessageTemplate,
        host: str,
        model: Optional[str] = None,
        token: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            template (MessageTemplate): The template used when calling llm.
            host (str): API of the llm, ex: "https://api.openai.com/v1", "http://192.168.10.10/v1".
            model (str, defaults to `None`): The name of llm, ex: "gpt-3.5-turbo", "gpt-4-turbo". It is a useless parameter for some hosts (ex: TGI)
            token (str, defaults to `None`): API token.
            generation_config (Optional[Dict[str, Any]], defaults to `None`): llm generation parameters. There are different settings according to different hosts. 
        """
        self.host = host
        self.template = template
        self.model = model
        self.token = token
        self.generation_config = copy.deepcopy(generation_config) if generation_config != None else {}
        
    def __call__(
        self, 
        inputs: Optional[Dict[str, str]] = None,
    ):
        """
        Fill in the input into the template and get the response from llm

        Args:
            inputs (Dict[str, str]): Parameters to be put in self.template, ex : {"query": "蘋果是甚麼顏色?"}.
        """
        headers = { "Authorization": f"Bearer {self.token}"}
        messages = self.template.format(inputs)
        data = {
            "model": self.model,
            "messages": messages,
        }
        data.update(self.generation_config)

        try:
            r = requests.post(self.host+"/chat/completions", json=data, headers=headers)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to make POST request: {e}")

        try:
            response = r.json()
            response_text = response["choices"][0]["message"]["content"]
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"Failed to parse response JSON: {e}\nresponse: {response}")

        return response_text