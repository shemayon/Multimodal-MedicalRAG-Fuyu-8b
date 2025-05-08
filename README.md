
# 🧠 Multimodal-MedicalRAG-Fuyu-8b

A lightweight, medical-focused Retrieval-Augmented Generation (RAG) system built using LangChain and the powerful fuyu-8b model. This project enables conversational medical question answering using fine-tuned instruction-output datasets and efficient 4-bit inference for resource-constrained environments.

🚑 Use Case

Designed to assist with:

* Medical Q\&A from dialogue datasets
* Rapid decision support from large document corpora
* Triage simulation and general patient concerns (non-diagnostic)

🧬 Architecture Overview

💬 Input Question
⟶ 🔍 Retrieve Top-k Relevant Chunks
⟶ 🧠 Generate Answer via fuyu-8b
⟶ ✅ Response Output

Powered by:

* fuyu-8b-sharded (via HuggingFace Transformers)
* LangChain's RetrievalQA
* ChromaDB for vector storage
* sentence-transformers for embeddings

🧪 Sample Prompt

```python
"My daughter has been experiencing POTS-like symptoms, and her heart rate is around 170. Should I take her to the ED, call an on-call nurse, or wait for her upcoming cardiology appointment?"
```

📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

📁 Dataset

We use the knowrohit07/know\_medical\_dialogue\_v2 dataset from HuggingFace for simulating doctor-patient conversations.

🔧 Local Deployment

Clone the repository:

```bash
git clone https://github.com/shemayon/Multimodal-MedicalRAG-Fuyu-8b.git
cd Multimodal-MedicalRAG-Fuyu-8b
```

Run the script:

```bash
python main.py
```

GPU strongly recommended for fuyu-8b inference (tested on NVIDIA A100, RTX 3090).

🧠 Model Details

* Embedding: sentence-transformers/all-MiniLM-L6-v2
* LLM: ybelkada/fuyu-8b-sharded (with 4-bit quantization)
* Vector Store: ChromaDB

📌 Notes

* This tool is research-grade. Always consult licensed professionals for actual medical advice.
* Prompt engineering and chunking strategies can significantly affect the system’s behavior.

💡 Future Ideas

* Integrate multimodal support (Fuyu can handle vision+text)
* Expand dataset with real patient records (anonymized)
* Add LangChain Agentic tools for tool-augmented diagnostics


## 🛡️ License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🌐 Connect

**Project by [Shemayon Soloman](http://www.linkedin.com/in/shemayon-soloman-b32387218)**  
Follow for updates, roadmap discussions, and launch announcements!



