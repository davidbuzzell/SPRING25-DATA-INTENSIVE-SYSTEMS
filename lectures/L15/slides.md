---
title: MBAI 417
separator: <!--s-->
verticalSeparator: <!--v-->
theme: serif
revealOptions:
  transition: 'none'
---

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 50%; position: absolute;">

  # Data Intensive Systems
  ## L.15 | R.A.G. & M.C.P.

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 10%">

  <iframe src="https://lottie.host/embed/216f7dd1-8085-4fd6-8511-8538a27cfb4a/PghzHsvgN5.lottie" height = "100%" width = "100%"></iframe>
  </div>
</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Welcome to Data Intensive Systems.
  ## Please check in by creating an account and entering the provided code.

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 40%; padding-top: 5%">
    <iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=Check In" width = "100%" height = "100%"></iframe>
  </div>
</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how comfortable are you with topics like:

  1. Retrieval-Augmented Generation (RAG)
  2. Streamlit GUIs
  3. Model Context Protocol (MCP)

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class="header-slide">

# Retrieval Augmented Generation (RAG)

</div>

<!--s-->

## Retrieval Augmented Generation (RAG)

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Chunking

Split text into smaller chunks for efficient retrieval.

### Indexing
Create an index of chunks for fast retrieval.

</div>
<div class="c2" style = "width: 50%">

### Retrieval
Retrieve relevant chunks based on the input query.

### Generation
Generate text based on the retrieved chunks and the input query.

</div>
</div>

<!--s-->

## Motivation | RAG

Large language models (LLMs) have revolutionized natural language processing (NLP) by achieving state-of-the-art performance on a wide range of tasks. We will discuss LLMs in more detail later in this lecture. However, for now it's important to note that modern LLMs have some severe limitations, including:

- **Inability to (natively) access external knowledge**
- **Hallucinations** (generating text that is not grounded in reality)

Retrieval-Augmented Generation (RAG) is an approach that addresses these limitations by combining the strengths of information retrieval systems with LLMs.

<!--s-->

## Motivation | RAG

So what is Retrieval-Augmented Generation (RAG)?

1. **Retrieval**: A storage & retrieval system that obtains context-relevant documents from a database.
2. **Generation**: A large language model that generates text based on the obtained documents.

<img src = "https://developer-blogs.nvidia.com/wp-content/uploads/2023/12/rag-pipeline-ingest-query-flow-b.png" style="margin: 0 auto; display: block; width: 80%; border-radius: 10px;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">NVIDIA, 2023</span>

<!--s-->

## Motivation | Creating an Expert Chatbot 🤖

<div style = "font-size: 0.8em;">

Our goal today is to build a RAG system that will answer questions about Northwestern's policy on academic integrity. To do this, we will:

1. **Chunk** the document into smaller, searchable units.<br>
Chunking is the process of creating windows of text that can be indexed and searched. We'll learn how to chunk text to make it compatible with a vector database.

2. **Embed** the text chunks.<br>
Word embeddings are dense vector representations of words that capture semantic information. We'll learn how to embed chunks using OpenAI's embedding model (and others!).

3. **Store and Retrieve** the embeddings from a vector database.<br>
We'll store the embeddings in a vector database and retrieve relevant documents based on the current context of a conversation. We'll demo with chromadb.

4. **Generate** text using the retrieved chunks and conversation context.<br>
We'll generate text with GPT-4 based on the retrieved chunks and a provided query, using OpenAI's API.

</div>

<!--s-->

<div class="header-slide">

# Chunk

</div>

<!--s-->

## Chunk | 🔥 Tips & Methods

<div style = "font-size: 0.8em;">

Chunking is the process of creating windows of text that can be indexed and searched. Chunking is essential for information retrieval systems because it allows us to break down large documents into smaller, searchable units.

<div class = "col-wrapper">

<div class="c1" style = "width: 50%; height: 100%; margin-right: 2em;">


### Sentence Chunking

Sentence chunking is the process of breaking text into sentences.

E.g. <span class="code-span">"Hello, world! How are you?" -> ["Hello, world!", "How are you?"]</span>

### Paragraph Chunking

Paragraph chunking is the process of breaking text into paragraphs.

E.g. <span class="code-span">"Hello, world! \n Nice to meet you." -> ["Hello, world!", "Nice to meet you."]</span>

### Agent Chunking

Agent chunking is the process of breaking text down using an LLM.

</div>

<div class="c2" style = "width: 50%; height: 100%;">

### Sliding Word / Token Window Chunking

Sliding window chunking is a simple chunking strategy that creates windows of text by sliding a window of a fixed size over the text.

E.g. <span class="code-span">"The cat in the hat" -> ["The cat in", "cat in the", "in the hat"]</span>

### Semantic Chunking

Semantic chunking is the process of breaking text into semantically meaningful units.

E.g. <span class="code-span">"The cat in the hat. One of my favorite books." -> ["The cat in the hat.", "One of my favorite books."]</span>

</div>
</div>

<!--s-->

## Chunk | NLTK Sentence Chunking

NLTK is a powerful library for natural language processing that provides many tools for text processing. NLTK provides a sentence tokenizer that can be used to chunk text into sentences.

### Chunking with NLTK

```python
from nltk import sent_tokenize

# Load Academic Integrity document.
doc = open('/Users/joshua/Desktop/academic_integrity.md').read()

# Split the document into sentences.
chunked_data = sent_tokenize(doc)
```

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; height: 100%;">

### Input: Original Text

```text
The purpose of this guide is to set forth the terms under which academic work is pursued at Northwestern and
throughout the larger intellectual community of which we are members. Please read this booklet carefully,
as you will be held responsible for its contents. It describes the ways in which common sense and decency apply
to academic conduct. When you applied to Northwestern, you agreed to abide by our principles of academic integrity;
these are spelled out on the first three pages. The balance of the booklet provides information that will help you avoid
violations, describes procedures followed in cases of alleged violations of the guidelines, and identifies people who 
can give you further information and counseling within the undergraduate schools.
```

</div>
<div class="c2" style = "width: 50%; height: 100%;">

### Output: Chunked Text (by Sentence)
```text
[
    "The purpose of this guide is to set forth the terms under which academic work is pursued at Northwestern and throughout the larger intellectual community of which we are members."
    "Please read this booklet carefully, as you will be held responsible for its contents."
    "It describes the ways in which common sense and decency apply to academic conduct."
    "When you applied to Northwestern, you agreed to abide by our principles of academic integrity; these are spelled out on the first three pages."
    "The balance of the booklet provides information that will help you avoid violations, describes procedures followed in cases of alleged violations of the guidelines, and identifies people who can give you further information and counseling within the undergraduate schools."
]

```
</div>
</div>

<!--s-->

<div class="header-slide">

# Embed

</div>

<!--s-->

## Embed

Word embeddings are dense vector representations of words that capture semantic information. Word embeddings are essential for many NLP tasks because they allow us to work with words in a continuous and meaningful vector space.

**Traditional embeddings** such as Word2Vec are static and pre-trained on large text corpora.

**Contextual embeddings** such as those produced by BERT (encoder-only Transformer model) are dynamic and depend on the context in which the word appears. Contextual embeddings are essential for many NLP tasks because they capture the *contextual* meaning of words in a sentence.

<img src="https://miro.medium.com/v2/resize:fit:2000/format:webp/1*SYiW1MUZul1NvL1kc1RxwQ.png" style="margin: 0 auto; display: block; width: 80%; border-radius: 10px;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Google</span>

<!--s-->

## Embed | Contextual Word Embeddings

Contextual word embeddings are word embeddings that are dependent on the context in which the word appears. Contextual word embeddings are essential for many NLP tasks because they capture the *contextual* meaning of words in a sentence.

For example, the word "bank" can have different meanings depending on the context:

- **"I went to the bank to deposit my paycheck."**
- **"The river bank was covered in mud."**

[HuggingFace](https://huggingface.co/spaces/mteb/leaderboard) contains a [MTEB](https://arxiv.org/abs/2210.07316) leaderboard for some of the most popular contextual word embeddings:

<img src="https://storage.googleapis.com/cs326-bucket/lecture_14/leaderboard.png" style="margin: 0 auto; display: block; width: 50%;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">HuggingFace, 2024</span>
<!--s-->

## Embed | OpenAI's Embedding Model

OpenAI provides an embedding model via API that can embed text into a dense vector space. The model is trained on a large text corpus and can embed text into a n-dimensional vector space.

```python
import openai

openai_client = openai.Client(api_key = os.environ['OPENAI_API_KEY'])
embeddings = openai_client.embeddings.create(model="text-embedding-3-large", documents=chunked_data)
```

🔥 Although they do not top the MTEB leaderboard, OpenAI's embeddings work well and the convenience of the API makes them a popular choice for many applications.

<!--s-->

<div class="header-slide">

# Retrieve

</div>

<!--s-->

## Store & Retrieve

A vector database is a database that stores embeddings and allows for fast similarity search. Vector databases are essential for information retrieval systems because they enable us to *quickly* retrieve relevant documents based on their similarity to a query. 

This retrieval process is very similar to a KNN search! However, vector databases will implement Approximate Nearest Neighbors (ANN) algorithms to speed up the search process -- ANN differs from KNN in that it does not guarantee the exact nearest neighbors, but rather a set of approximate nearest neighbors.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

There are many vector databases options available, such as:

- [ChromaDB](https://www.trychroma.com/)
- [Pinecone](https://www.pinecone.io/product/)
- [Vector Search](https://cloud.google.com/vertex-ai/docs/vector-search/overview)
- [Postgres with PGVector](https://github.com/pgvector/pgvector)
- [FAISS](https://ai.meta.com/tools/faiss/)
- ...

</div>
<div class="c2" style = "width: 50%">

<img src = "https://miro.medium.com/v2/resize:fit:1400/format:webp/1*bg8JUIjbKncnqC5Vf3AkxA.png" style="margin: 0 auto; display: block; width: 80%;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Belagotti, 2023</span>

</div>
</div>

<!--s-->

## Store & Retrieve | ChromaDB

<div style="font-size: 0.9em">

ChromaDB is a vector database that stores embeddings and allows for fast text similarity search. ChromaDB is built on top of SQLite and provides a simple API for storing and retrieving embeddings.

### Initializing

Before using ChromaDB, you need to initialize a client and create a collection.

```python
import chromadb
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection('academic_integrity_nw')
```

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Storing Embeddings

Storing embeddings in ChromaDB is simple. You can store embeddings along with the original documents and ids.

```python
# Store embeddings in chromadb.
collection.add(embeddings = embeddings, documents = chunked_data, ids = [f"id.{i}" for i in range(len(chunked_data))])
```

</div>
<div class="c2" style = "width: 50%">

### Retrieving Embeddings

You can retrieve embeddings from ChromaDB based on a query. ChromaDB will return the most similar embeddings (and the original text) to the query.

```python
# Get relevant documents from chromadb, based on a query.
query = "Can a student appeal?"
relevant_chunks = collection.query(query_embeddings = embedding_function([query]), n_results = 2)['documents'][0]

>>> ['A student may appeal any finding or sanction as specified by the school holding jurisdiction.',
     '6. Review of any adverse initial determination, if requested, by an appeals committee to whom the student has access in person.']

```

</div>
</div>
</div>

<!--s-->

## Retrieving Embeddings | 🔥 Tips & Re-Ranking

In practice, the retrieved documents may not be in the order you want. While a vector db will often return documents in order of similarity to the query, you can re-rank documents based on a number of factors. Remember, your chatbot is paying per-token on calls to LLMs. You can cut costs by re-ranking the most relevant documents first and only sending those to the LLM.

<div class = "col-wrapper">

<div class="c1" style = "width: 50%; margin-right: 2em;">

### Multi-criteria Optimization

Consideration of additional factors beyond similarity, such as document quality, recency, and 'authoritativeness'.

### User Feedback

Incorporate user feedback into the retrieval process. For example, if a user clicks on a document, it can be re-ranked higher in future searches.

</div>

<div class="c2" style = "width: 50%">

### Diversification

Diversify the search results by ensuring that the retrieved documents cover a wide range of topics.

### Query Expansion & Rephrasing

For example, if a user asks about "academic integrity", the system could expand the query to include related terms like "plagiarism" and "cheating". This will help retrieve more relevant documents.

<!--s-->

<div class="header-slide">

# Generate

</div>

<!--s-->

## Generate

Once we have retrieved the relevant chunks based on a query, we can generate text using a large language model. Large language models can be used for many tasks -- including text classification, text summarization, question-answering, multi-modal tasks, and more.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

There are many large language models available at platforms like:

- [OpenAI GPT-4o](https://platform.openai.com/)
- [Google Gemini](https://ai.google.dev/gemini-api/docs?gad_source=1&gclid=CjwKCAiAudG5BhAREiwAWMlSjKXwuvq9JRRX0xxXaS7yCSn-NWo3e4rso3D-enl2IblIH09phtCvSxoCJhoQAvD_BwE)
- [Anthropic Claude](https://claude.ai/)
- [HuggingFace (Many)](https://huggingface.co/)
- ...


</div>
<div class="c2" style = "width: 50%">

<img src="https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcce3c437-4b9c-4d15-947d-7c177c9518e5_4258x5745.png" style="margin: 0 auto; display: block; width: 80%;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Raschka, 2023</span>

</div>
</div>

<!--s-->

## Generate | GPT-4 & OpenAI API

What really sets OpenAI apart is their extremely useful and cost-effective API. This puts their LLM in the hands of users with minimal effort. Competitors liuke Anthropic and Google have similar APIs now.

```python

import openai

openai_client = openai.Client(api_key = os.environ['OPENAI_API_KEY'])
response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi, GPT-4!"}
    ]
)

```

<!--s-->

## Generate | Prompt Engineering 🔥 Tips

<div class = "col-wrapper" style = "font-size: 0.9em;">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Memetic Proxy

A memetic proxy is a prompt that provides context to the LLM by using a well-known meme or phrase. This can help the LLM derive the context of the conversation and generate more relevant responses. [McDonell 2021](https://arxiv.org/pdf/2102.07350).

### Few-Shot Prompting
Few-shot prompting is a technique that provides the LLM with a few examples of the desired output. This can help the LLM understand the context and generate more relevant responses. [OpenAI 2023](https://arxiv.org/abs/2303.08774).

</div>
<div class="c2" style = "width: 50%">

### Chain-of-Thought Prompting
Chain-of-thought prompting is a technique that provides the LLM with a series of steps to follow in order to generate the desired output. This can help the LLM understand the context and generate more relevant responses. [Ritter 2023](https://arxiv.org/pdf/2305.14489).

### **TYPOS AND CLARITY**
Typos and clarity are important factors to consider when generating text with an LLM. Typos can lead to confusion and misinterpretation of the text, while clarity can help the LLM with the context and generate more relevant responses.

</div>
</div>

<!--s-->

<div class="header-slide">

# Putting it All Together

</div>

<!--s--> 

## Putting it All Together

Now that we have discussed the components of Retrieval-Augmented Generation (RAG), let's use what we have learned to build an expert chatbot that can answer questions about Northwestern's policy on academic integrity.

<img src = "https://developer-blogs.nvidia.com/wp-content/uploads/2023/12/rag-pipeline-ingest-query-flow-b.png" style="margin: 0 auto; display: block; width: 80%; border-radius: 10px;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">NVIDIA, 2023</span>

<!--s-->

## Putting it All Together | Demo Copied Here

```python[1-10 | 12-13 | 15-16 | 18-19 | 21-23 | 25-26 | 28 - 39 | 40 - 42 | 44-46 | 48 - 51]
import os

import chromadb
import openai
from chromadb.utils import embedding_functions
from nltk import sent_tokenize

# Initialize clients.
chroma_client = chromadb.Client()
openai_client = openai.Client(api_key = os.environ['OPENAI_API_KEY'])

# Create a new collection.
collection = chroma_client.get_or_create_collection('academic_integrity_nw')

# Load academic integrity document.
doc = open('/Users/joshua/Documents/courses/SPRING25-GENERATIVE-AI/docs/academic_integrity.md').read()

# Chunk the document into sentences.
chunked_data = sent_tokenize(doc)

# Embed the chunks.
embedding_function = embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-ada-002", api_key=os.environ['OPENAI_API_KEY'])
embeddings = embedding_function(chunked_data)

# Store embeddings in ChromaDB.
collection.add(embeddings = embeddings, documents = chunked_data, ids = [f"id.{i}" for i in range(len(chunked_data))])

# Create a system prompt template.

SYSTEM_PROMPT = """

You will provide a response to a student query using exact language from the provided relevant chunks of text.

RELEVANT CHUNKS:

{relevant_chunks}

"""

# Get user query.
user_message = "Can I appeal?"
print("User: " + user_message)

# Get relevant documents from chromadb.
relevant_chunks = collection.query(query_embeddings = embedding_function([user_message]), n_results = 2)['documents'][0]
print("Retrieved Chunks: " + str(relevant_chunks))

# Send query and relevant documents to GPT-4.
system_prompt = SYSTEM_PROMPT.format(relevant_chunks = "\n".join(relevant_chunks))
response = openai_client.chat.completions.create(model="gpt-4", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}])
print("RAG-GPT Response: " + response.choices[0].message.content)

```

```text
User: Can a student appeal?
Retrieved Chunks: ['A student may appeal any finding or sanction as specified by the school holding jurisdiction.', '6. Review of any adverse initial determination, if requested, by an appeals committee to whom the student has access in person.']
RAG-GPT Response: Yes, a student may appeal any finding or sanction as specified by the school holding jurisdiction.
```

<!--s-->

<div class="header-slide">

# Wrapping RAG in a pretty GUI
## Streamlit App

</div>

<!--s-->

<div class="header-slide">

# Model Context Protocol

</div>

<!--s-->

## Model Context Protocol (MCP)

"MCP is an open protocol that standardizes how applications provide context to LLMs. Think of MCP like a USB-C port for AI applications. Just as USB-C provides a standardized way to connect your devices to various peripherals and accessories, MCP provides a standardized way to connect AI models to different data sources and tools." 

<div style="text-align: right; margin-right: 2em;">
   <a href="https://docs.anthropic.com/en/docs/agents-and-tools/mcp">Anthropic, 2025</a>
</div>

<!--s-->

## Model Context Protocol (MCP) | Overview

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/mcp.png' style='border-radius: 10px; width: 70%;'>
   <p style='font-size: 0.6em; color: grey;'>Anthropic 2025</p>
</div>

<!--s-->

<div class="header-slide">

# Demo of MCP
## [File System](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem) & Claude

</div>

<!--s-->

<div class="header-slide">

# Demo of BYO-MCP
## Weather API Integration

</div>

<!--s-->

<div class="header-slide">

# Demo of MCP
## Spotify API & Claude

</div>

<!--s-->

## MCP Note

MCP is a push from Anthropic to establish a standard for how LLMs can interact with external data sources and tools. It is under active development! Here is a post from 05.01.2025 that describes using [remote MCP servers](https://www.anthropic.com/news/integrations).

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how comfortable are you with topics like:

  1. Retrieval-Augmented Generation (RAG)
  2. Streamlit GUIs
  3. Model Context Protocol (MCP)

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->