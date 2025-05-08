
# import packages
from langchain.document_loaders import DataFrameLoader
from langchain.vectorstores import Chroma
from datasets import load_dataset
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import  pipeline
from langchain.text_splitter import CharacterTextSplitter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
model_name = "ybelkada/fuyu-8b-sharded"
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.bos_token_id = 1
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=2048,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.2
)

local_llm = HuggingFacePipeline(pipeline=pipe)


train_dataset = load_dataset("knowrohit07/know_medical_dialogue_v2" , split="train")
train_dataset = train_dataset.to_pandas()
train_dataset['text'] = train_dataset["instruction"] +  train_dataset["output"]
df_document = DataFrameLoader(train_dataset[:1000] , page_content_column="text").load()
text_splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=10)
texts = text_splitter.split_documents(df_document)


chromadb_index = Chroma.from_documents(texts, embed_model , persist_directory="DB")
local_llm("I've been experiencing chest thumps, and the ER doctor diagnosed PVCs. Is this normal, and should I seek a cardiology referral?	")

document_qa = RetrievalQA.from_chain_type(
    llm=local_llm, chain_type="stuff", retriever=chromadb_index.as_retriever(search_kwargs={"k": 5})
)

torch.cuda.empty_cache()

response = document_qa.run("My daughter has been experiencing POTS-like symptoms, and her heart rate is around 170. Should I take her to the ED, call an on-call nurse, or wait for her upcoming cardiology appointment in March?")
print(response)