# 🤖 Chatbot with LangChain, Pinecone, and Hugging Face

A chatbot that automatically creates an index, retrieves data from Hugging Face (MS MARCO), and builds a **LangChain RAG (Retrieval-Augmented Generation) system** using **Pinecone** for embeddings and **Hugging Face** for the language model.


## 🚀 How to Run

### 1️⃣ **Build the Docker Image** (Takes ~10 minutes)
```sh
sudo docker-compose build
```

### 2️⃣ **Run the Chatbot in Interactive Mode**
```sh
sudo docker-compose run --rm chatbot
```

**Chatbot Response Time:** *~1 minute per question*

**First Time Loading Dataset Time** *~5 minutes, 1000 documents added to index*

**Note:** The chatbot runs interactively and expects user input. Press **Ctrl+C** to stop it.


## 📜 Example Output

```
$ sudo docker-compose run --rm chatbot
Creating rag-langchain_chatbot_run ... done
Connecting to vector store
Connecting to llm
Ready!
The bot will answer questions based on MSMarco dataset. Type 'exit' or 'q' to exit.
Try with questions based on MSMarco: What is RBA?

Question: What is RBA?
Answer: RBA stands for Results-Based Accountability, a methodology that guides communities and organizations in enhancing the well-being of various groups through collective improvement efforts.
```

**Note:** If the **Hugging Face free tier is overloaded**, responses may be unavailable temporarily. Wait a few minutes and try again.


## 📌 Required `.env` File
Before running, create a `.env` file in the root directory with:
```ini
PINECONE_API_KEY=your_pinecone_api_key_here
HUGGING_FACE_API_KEY=your_hugging_face_api_key_here
```

**Note:** When creating Hugging Face API key, enable inference permissions. Other permissions can be disabled.

**Warning:** Do **NOT** share your API keys publicly!


## 🛠 How It Works

1. **Creates an Index** - The app automatically sets up a **Pinecone index**.
2. **Fills with Data** - It retrieves **MS MARCO data** from Hugging Face and embeds it.
3. **Builds LangChain RAG** - Uses **Pinecone** for retrieval and **Hugging Face LLM** for responses.
4. **Interacts via Terminal** - Users can enter queries, and the chatbot provides answers.


## 📝 Author
- **Goran Ivankovic**
- 🔗 GitHub: [ranoiv](https://github.com/rangoiv)

