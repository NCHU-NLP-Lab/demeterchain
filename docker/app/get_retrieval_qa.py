import os
from pathlib import Path
from demeterchain.retrievers import PyseriniBM25Retriever
from demeterchain.models import GenerativeModel
from demeterchain.utils import QAModelConfig, PromptTemplate
from demeterchain.utils import QAConfig
from demeterchain.chains import RetrievalQA

CACHE_DIR = "/model"
DATA_FILEPATH = "/data"
RETIREVER_FILEPATH = "pyserini_bm25_index"

if 'HUGGING_FACE_HUB_TOKEN' in os.environ:
    from huggingface_hub import login
    login(os.environ['HUGGING_FACE_HUB_TOKEN'])
    

def get_retriever():
    data_dict = Path(DATA_FILEPATH)
    retriever_dict = data_dict / RETIREVER_FILEPATH
    if retriever_dict.exists() and retriever_dict.is_dir():
        retriever = PyseriniBM25Retriever.load(retriever_dict)
    else:
        from demeterchain.loaders import TextLoader
        from demeterchain.splitters import TextSplitter

        loader = TextLoader(data_dict, glob='*.txt', show_progress=True)
        documents = loader.load()
        retriever = PyseriniBM25Retriever.from_documents(documents, savepath=retriever_dict)
    
    return retriever

def get_reader(args):
    model_config = QAModelConfig(
        model_name_or_path = args.model_name_or_path,
        device_map = args.device_map,
        cache_dir = CACHE_DIR,
        quantize = args.quantize,
        dtype = args.quantize,
        use_flash_attention = args.use_flash_attention,
        noanswer_str = "無法回答。",
        template = PromptTemplate(
            input_variables = ["doc", "query"],
            template="[INST] <<SYS>>\n請根據提供的問題，從提供的內文中尋找答案並回答，回答時只需要輸出答案，不需輸出其他資訊，如果從提供的內文無法找到答案，請回答\"無法回答\"\n<</SYS>>\n\n問題:\n{query}\n\n內文:\n{doc}\n [/INST]答案:\n"
        )
    )
    reader = GenerativeModel(config=model_config)

    return reader

def get_retrieval_qa(args):
    retriever = get_retriever()
    reader = get_reader(args)
    retrieval_qa = RetrievalQA(
        reader=reader, 
        retriever=retriever, 
    )
    qa_config = QAConfig(retrieve_k = args.retrieve_k)
    return retrieval_qa, qa_config

