import copy
from typing import Optional, Dict
from demeterchain.utils import QAConfig, QAResult
from demeterchain.models import BaseModel
from demeterchain.llms import ChatCompletionsLLM


class RetrievalQA:
    """
    Connect retriever and qamodel in series

    Examples:

        .. code-block:: python
            from demeterchain.models import GenerativeModel
            from demeterchain.utils import PromptTemplate, MessageTemplate, QAModelConfig, QAConfig
            from demeterchain.llms import ChatCompletionsLLM
            from demeterchain.chains import RetrievalQA
            from demeterchain.retrievers import PyseriniBM25Retriever

            # load retriever
            retriever = PyseriniBM25Retriever.load("/path/to/retriever")

            # load model
            model_config = QAModelConfig(
                model_name_or_path = "/path/to/model",
                noanswer_str = "無法回答",
                template = PromptTemplate(
                    input_variables = ["doc", "query"],
                    template="[INST] <<SYS>>\n請根據提供的問題，從提供的內文中尋找答案並回答，回答時只需要輸出答案，不需輸出其他資訊，如果從提供的內文無法找到答案，請回答\"無法回答\"\n<</SYS>>\n\n問題:\n{query}\n\n內文:\n{doc}\n [/INST]答案:\n"
                )
            )
            reader = GenerativeModel(config=model_config)

            # set hyde and summary llm
            token = "your_token" # Please fill in your token
            host = "https://api.openai.com/v1"
            model = "gpt-4-turbo"

            hyde_template = MessageTemplate(
                input_variables = ["query"],
                template=[
                    {"role": "system", "content": "請幫我解答以下問題，請用中文回答。",},
                    {"role": "user", "content": "{query}"}
                ]
            )
            
            hyde_llm = ChatCompletionsLLM(
                host = host,
                model = model,
                token = token,
                template= hyde_template,
                generation_config={
                    "max_tokens": 200,
                    "temperature": 0.9,
                }
            )

            summary_template = MessageTemplate(
                input_variables = ["query", "answers"],
                template=[
                    {"role": "system", "content": "請幫完成任務以下問題，請用中文回答。",},
                    {"role": "user", "content": "我有以下片段關於{query}的答案：{answers} 你可以試著幫我將結果進行統整摘要嗎？"}
                ]
            )
            summary_llm = ChatCompletionsLLM(
                host = host,
                model = model,
                token = token,
                template= summary_template,
                generation_config={
                    "max_tokens": 600,
                    "temperature": 0.9,
                }
            )

            # Connect all components
            qa = RetrievalQA(
                reader=reader, 
                retriever=retriever, 
                hyde_llm=hyde_llm, # If not set, this function will not be activated
                summary_llm=summary_llm # If not set, this function will not be activated
            )

            # get answer
            qa_config = QAConfig(retrieve_k = 6)
            answer, answer_with_docs = qa({"query": "水稻稻熱病的可能成因？"}, qa_config=qa_config)

            print(answer)
    """
    def __init__(
        self, 
        reader: BaseModel, 
        retriever,  
        hyde_llm: Optional[ChatCompletionsLLM] = None,
        summary_llm: Optional[ChatCompletionsLLM] = None
    ):
        self.reader=reader
        self.retriever=retriever
        self.hyde_llm = hyde_llm
        self.summary_llm = summary_llm
    
    def __call__(
        self, 
        inputs: Dict[str, str],
        qa_config: QAConfig,
    ):
        """
        Get answers to input questions

        Args:
            inputs (Dict[str, str]): Parameters to be put in self.template, ex : {"query": "什麼水果是紅色的?"}.
            qa_config (QAConfig): Parameter settings for retrieving and generating answers.
            
        """
        if self.hyde_llm != None:
            query = inputs["query"] + " " + self.hyde_llm(inputs)
        else:
            query = inputs["query"]
            
        docs = self.retriever.invoke(query, k = qa_config.retrieve_k)
        answer_docs = self.reader.get_answer(
            inputs, 
            docs, 
            batch_size = qa_config.batch_size, 
            max_length = qa_config.max_length,
            max_new_tokens = qa_config.max_new_tokens,
            num_beams = qa_config.num_beams,
            num_return_sequences = qa_config.num_return_sequences,
            answer_strategy = qa_config.answer_strategy,
        )

        if len(answer_docs) == 0:
            return QAResult(answers={'很抱歉，模型無法根據現有資料集回答您的問題。': None}) 
        
        answers = list(answer_docs.keys())

        summary = None
        if self.summary_llm != None:
            summary_inputs = copy.deepcopy(inputs)
            summary_inputs.update({"answers": str(answers)})
            summary = self.summary_llm(summary_inputs)
        
        return QAResult(summary=summary, answers=answer_docs)
