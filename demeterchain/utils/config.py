import torch
from typing import Union, Optional, List
from demeterchain.utils import PromptTemplate, MessageTemplate


class QAModelConfig(object):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `demeterchain`.

    Currently only supports `LLM.int8()`and `NF4` quantization. 

    Args:
        model_name_or_path (str): 
            Can be either:
                - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                - A path to a *directory* containing model weights saved using
                    [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
        template (Union[PromptTemplate, MessageTemplate]): template used by the model.
        device_map (str, defaults to `auto`): The device on which the model will be placed
        dtype (Union[str, torch.dtype], defaults to torch.float16): dtype used by the model
        quantize (Optional[str], defaults to `None`): 
            The quantification method used by the model.
            Can be either:
                - bitsandbytes: `LLM.int8()`
                - bitsandbytes-nf4: `NF4`
        use_flash_attention (bool, defaults to `False`): Whether to enable flash_attention_2.
        cache_dir (Optional[str], defaults to `None`): Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used.
        noanswer_str (Optional[str], defaults to `None`): String which let the model decide whether to ignore the document. 
        noanswer_ids (Optional[List[int]], defaults to `None`): tokenized String which let the model decide whether to ignore the document. 
    """

    ALLOWED_QUANTIZE_TYPE = ["bitsandbytes", "bitsandbytes-nf4"]
    
    def __init__(
        self, 
        model_name_or_path: str,
        template: Union[PromptTemplate, MessageTemplate],
        device_map: str = "auto",
        dtype: Union[str, torch.dtype] = torch.float16,
        quantize: Optional[str] = None,
        use_flash_attention: bool = False,
        cache_dir: Optional[str] = None,
        noanswer_str: Optional[str] = None,
        noanswer_ids: Optional[List[int]] = None,
    ):
        self.model_name_or_path=model_name_or_path
        self.template=template
        self.device_map=device_map
        self.use_flash_attention = use_flash_attention
        self.cache_dir = cache_dir
        self.noanswer_str=noanswer_str
        self.noanswer_ids=noanswer_ids

        if isinstance(dtype, torch.dtype):
            self.dtype = dtype
        elif dtype == "float32":
            self.dtype = torch.float32
        elif dtype == "float16":
            self.dtype = torch.float16
        elif dtype == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float16
        self.quantize = quantize if quantize in self.ALLOWED_QUANTIZE_TYPE else None
    
    def __repr__(self):
        config_info = (
            "QAModelConfig("
            f"model_name_or_path={self.model_name_or_path}, "
            f"device_map={self.device_map}, "
            f"dtype={self.dtype}, "
            f"use_flash_attention={self.use_flash_attention}, "
            f"cache_dir={self.cache_dir}, "
            f"noanswer_str={self.noanswer_ids}, "
            f"template={self.template}"
            f")")
        return config_info


class QAConfig:
    ALLOWED_ANSWER_STRATEGY = ["best", "longest"]
    def __init__(
        self, 
        retrieve_k: int = 20,
        batch_size: int = 1,
        max_length: int = 768,
        max_new_tokens: int = 32,
        num_beams:int = 3,
        answer_strategy:str = "best",
    ):
        self.retrieve_k = retrieve_k
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.num_return_sequences = num_beams
        self.answer_strategy = answer_strategy if answer_strategy in self.ALLOWED_ANSWER_STRATEGY else "best"