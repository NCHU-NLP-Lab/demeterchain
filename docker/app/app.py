import argparse
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from get_retrieval_qa import get_retrieval_qa

app = FastAPI()
retrieval_qa, qa_config = None, None


class QueryItem(BaseModel):
    query: str

@app.post("/qa")
async def process_string(query_item: QueryItem):
    answers = retrieval_qa({"query": query_item.query}, qa_config=qa_config)
    results = [{"answer": answer.answer, "doc": answer.doc.page_content} for answer in answers.answers]
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FastAPI server.")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--model_name_or_path", type=str, default="NchuNLP/taide-qa", help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--device_map", type=str, default="auto", help="The device on which the model will be placed")
    parser.add_argument("--dtype", type=str, default="float16", help="dtype used by the model")
    parser.add_argument("--quantize", type=str, default=None, help="The quantification method used by the model.")
    parser.add_argument("--use_flash_attention", type=bool, default=False, help="Whether to enable flash_attention_2")
    parser.add_argument("--retrieve_k", type=int, default=3, help="The number of documents to be retrieved")
    args = parser.parse_args()

    retrieval_qa, qa_config = get_retrieval_qa(args)
    uvicorn.run(app, host="0.0.0.0", port=args.port)