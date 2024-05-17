import copy
from collections import OrderedDict
from typing import List, Dict, Optional


class PromptTemplate:
    """
    A class for filling in text templates with provided parameters.

    Examples:

        .. code-block:: python
            from demeterchain.utils import PromptTemplate

            template = PromptTemplate(
                input_variables = ["query"],
                template="請回答以下問題: {query}"
            )
            input_query = template.format({"query": "什麼水果是紅色的?"})
        >>> print(input_query)
        請回答以下問題: 什麼水果是紅色的?
    """

    def __init__(    
        self,
        input_variables: List[str],
        template: str
    ):
        '''
        Args:
            input_variables (List[str]): contains all variables that need to be entered, ex: ["var1", "var2"]
            template (str): string containing variables, curly brackets are required around the variables. ex: "hello {var1} world {var2}"
        '''
        input_variables = set(input_variables)
        variable_positions = []

        # check if the variable exists
        for variable in input_variables:
            position = template.find(f"{{{variable}}}")
            if position == -1:
                raise ValueError(
                    f"variable {variable} must be contained in template"
                )
            else:
                variable_positions.append((variable, position))
        
        # order variables to avoid repeated substitutions
        variable_positions = sorted(variable_positions, key=lambda x:x[1], reverse=True)

        variables = OrderedDict()
        for variable, position in variable_positions:
            variables[variable] = position

        self.variables = variables
        self.template = template

    def format(self, inputs: Dict[str, str]) -> str:
        prompt = self.template
        for variable in self.variables:
            if variable in inputs:
                prompt = prompt.replace(f"{{{variable}}}", inputs[variable])
            else:
                raise ValueError(
                    f"variable {variable} is missing"
                )
        return prompt

    def split_format(self, inputs: Dict[str, str], split_variable:str) -> (str, str):
        prompt_prefix = self.template
        prompt_postfix = ""
        for variable in self.variables:
            if variable == split_variable:
                prompt_prefix, prompt_postfix = prompt_prefix.split(f"{{{variable}}}", 1)
                continue
            if variable in inputs:
                prompt_prefix = prompt_prefix.replace(f"{{{variable}}}", inputs[variable])
            else:
                raise ValueError(
                    f"variable {variable} is missing"
                )
        return prompt_prefix, prompt_postfix


class MessageTemplate:
    """
    A class for filling in text templates with provided parameters.

    Examples:

        .. code-block:: python
            from demeterchain.utils import MessageTemplate

            template = MessageTemplate(
                input_variables = ["query"],
                template=[
                    {"role": "system", "content": "請幫我解答以下問題，請用中文回答。",},
                    {"role": "user", "content": "{query}"}
                ]
            )
            input_query = template.format({"query": "什麼水果是紅色的?"})
        >>> print(input_query)
        [{'role': 'system', 'content': '請幫我解答以下問題，請用中文回答。'}, {'role': 'user', 'content': '什麼水果是紅色的?'}]
    """
    def __init__(    
        self,
        template: List[Dict[str, str]],
        input_variables: Optional[List[str]] = None,
    ):
        '''
        Args:
            input_variables: contains all variables that need to be entered, ex: ["query", "doc"]
            template: message containing variables, curly brackets are required around the variables. 
                ex: [
                        {
                            "content": "you are a language model",
                            "role": "system"
                        },
                        {
                            "content": "Please answer the following questions according to the document. {query} {doc}",
                            "role": "user"
                        }
                    ]
        '''
        input_variables = input_variables if input_variables is not None else []
        if not isinstance(input_variables, list):
            raise TypeError(
                "input_variables is not a list.\n"
                "The following is a sample format\n"
                '`["query", "doc"]`.'
            )
        if not isinstance(template, list):
            raise TypeError(
                "template is not a list.\n"
                "The following is a sample format\n"
                '`[{"content": "you are a language model","role": "system"}, {"content": "Please answer the following questions according to the document. {query} {doc}","role": "user"}]`.'
            )


        input_variables = set(input_variables)
        variable_positions = []

        # check if the variable exists
        for variable in input_variables:
            # find all variables in template
            for i, message in enumerate(template):
                if "content" not in message or "role" not in message:
                    raise TypeError(
                        "message_template should contain \"content\" and \"role\".\n"
                        "The following is a sample format\n"
                        '`[{"content": "you are a language model","role": "system"}, {"content": "Please answer the following questions according to the document. {query} {doc}","role": "user"}]`.'
                    )
                
                position = message["content"].find(f"{{{variable}}}")
                if position != -1:
                    position = (i, position)
                    break
            if position == -1:
                raise ValueError(
                    f"variable {variable} must be contained in template"
                )
            else:
                variable_positions.append((variable, position))
        
        # order variables to avoid repeated substitutions
        variable_positions = sorted(variable_positions, key=lambda x:(x[1][0], x[1][1]), reverse=True)

        variables = OrderedDict()
        for variable, position in variable_positions:
            variables[variable] = position

        self.variables = variables
        self.template = copy.deepcopy(template)

    def format(self, inputs: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        inputs = inputs if inputs is not None else {}

        message = copy.deepcopy(self.template)
        for variable, pos in self.variables.items():
            if variable in inputs:
                _id = pos[0]
                message[_id]["content"] = message[_id]["content"].replace(f"{{{variable}}}", inputs[variable])
            else:
                raise ValueError(
                    f"variable {variable} is missing"
                )
        return message