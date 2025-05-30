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
  ## L.08 | EDA IV Searching Text

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

## Announcements

- H.03 will be released on Monday, April 28th.
- Exam Part I will take place on Monday, May 5th.

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you with **Text Mining** concepts such as:

  - Regular Expressions
  - Semantic Search w/ Embeddings
  - Visualizing Embeddings

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class="header-slide">

# EDA IV
## Searching Text

</div>

<!--s-->

## Agenda

- Regular Expressions (Searching for text patterns)
  - Syntax
  - Practice
  - Regex search in SnowFlake
- Word Embeddings (Searching for semantic meaning)
  - Traditional (word2vec)
  - Modern (LLMs)
  - Plotting embeddings
  - Semantic search in SnowFlake

<!--s-->

<div class="header-slide">

# Regular Expressions

</div>

<!--s-->

## Regular Expressions

Regular expressions (regex) are a powerful tool for working with text data. They allow us to search, match, and manipulate text using a concise and expressive syntax.

One may feel compelled to do basic sting manipulation with Python's built-in string methods. However, regular expressions are much more powerful and flexible. Consider the following example:

> "My phone number is (810) 555-1234."

<!--s-->

## Regular Expressions | Example

> "My phone number is (810)555-1234"

### String Methods

<span class="code-span">phone_number = text.split(" ")[-1]</span>

This method would work for the given example but **not** for "My phone number is (810) 555-1234. Call me!"
or "My phone number is (810) 555-1234. Call me! It's urgent!"

### Regular Expression

<span class="code-span">phone_number = re.search(r'\(\d{3}\)\d{3}-\d{4}', text).group()</span>

This regular expression will match any phone number in the format (810)555-1234, including the additional text above.

<!--s-->

## Regular Expressions | Syntax

Regular expressions are a sequence of characters that define a search pattern. They are used to search, match, and manipulate text strings.

<div style="font-size: 0.8em; overflow-y: scroll; height: 80%;">
  
| Pattern | Description |
|---------|-------------|
| <span class='code-span'>.</span>     | Matches any character except newline |
| <span class='code-span'>^</span>     | Matches the start of a string |
| <span class='code-span'>$</span>     | Matches the end of a string |
| <span class='code-span'>*</span>     | Matches 0 or more repetitions of the preceding element |
| <span class='code-span'>+</span>     | Matches 1 or more repetitions of the preceding element |
| <span class='code-span'>?</span>     | Matches 0 or 1 repetition of the preceding element |
| <span class='code-span'>{n}</span>   | Matches exactly n repetitions of the preceding element |
| <span class='code-span'>{n,}</span>  | Matches n or more repetitions of the preceding element |
| <span class='code-span'>{n,m}</span> | Matches between n and m repetitions of the preceding element |
| <span class='code-span'>[]</span>    | Matches any one of the characters inside the brackets |
| <span class='code-span'> \| </span>     | Matches either the expression before or the expression after the operator |
| <span class='code-span'>()</span>    | Groups expressions and remembers the matched text |
| <span class='code-span'>\d</span>    | Matches any digit (equivalent to <span class='code-span'>[0-9]</span>) |
| <span class='code-span'>\D</span>    | Matches any non-digit character |
| <span class='code-span'>\w</span>    | Matches any word character (equivalent to <span class='code-span'>[a-zA-Z0-9_]</span>) |
| <span class='code-span'>\W</span>    | Matches any non-word character |
| <span class='code-span'>\s</span>    | Matches any whitespace character (spaces, tabs, line breaks) |
| <span class='code-span'>\S</span>    | Matches any non-whitespace character |
| <span class='code-span'>\b</span>    | Matches a word boundary |
| <span class='code-span'>\B</span>    | Matches a non-word boundary |
| <span class='code-span'>\\</span>    | Escapes a special character |

</div>

<!--s-->

## Regular Expressions | Simple

| Pattern  | Description   | Example     | Matches   |
|----------|-----------|-------------|--------|
| <span class='code-span'>^abc</span>   | Matches string starting with <span class='code-span'>abc</span>     | <span class='code-span'>abcdef</span>    | <span class='code-span'>abc</span>          |
| <span class='code-span'>def$</span>   | Matches string ending with <span class='code-span'>def</span>       | <span class='code-span'>abcdef</span>    | <span class='code-span'>def</span>|
| <span class='code-span'>a.c</span>    | Matches <span class='code-span'>a</span> and <span class='code-span'>c</span> with any char between | <span class='code-span'>abc</span>, <span class='code-span'>a-c</span> | <span class='code-span'>abc</span>, <span class='code-span'>a-c</span>   |
| <span class='code-span'>a+</span>     | Matches one or more occurrences of <span class='code-span'>a</span> | <span class='code-span'>aaab</span>      | <span class='code-span'>aaa</span>    |
| <span class='code-span'>colou?r</span>| Matches <span class='code-span'>color</span> or <span class='code-span'>colour</span>   | <span class='code-span'>color</span>, <span class='code-span'>colour</span> | <span class='code-span'>color</span>, <span class='code-span'>colour</span>   |
| <span class='code-span'>[0-9]</span>  | Matches any digit   | <span class='code-span'>a1b2c3</span>    | <span class='code-span'>1</span>, <span class='code-span'>2</span>, <span class='code-span'>3</span>  |
| <span class='code-span'>[a-z]</span>  | Matches any lowercase letter  | <span class='code-span'>ABCabc</span>    | <span class='code-span'>a</span>, <span class='code-span'>b</span>, <span class='code-span'>c</span>  |
| <span class='code-span'>[^a-z]</span> | Matches any char not in set <span class='code-span'>a-z</span>       | <span class='code-span'>ABCabc123</span> | <span class='code-span'>A</span>, <span class='code-span'>B</span>, <span class='code-span'>C</span>, <span class='code-span'>1</span>, <span class='code-span'>2</span>, <span class='code-span'>3</span> |
| <span class='code-span'>\d</span>     | Matches any digit   | <span class='code-span'>123abc</span>    | <span class='code-span'>1</span>, <span class='code-span'>2</span>, <span class='code-span'>3</span>  |
| <span class='code-span'>\w+</span>    | Matches one or more word characters     | <span class='code-span'>Hello, world!</span> | <span class='code-span'>Hello</span>, <span class='code-span'>world</span>      |

<!--s-->

## Regular Expressions | Complex

| Pattern      | Description     | Example Input | Matches       |
|--------------|-------------------------------------------------|-----------------------|---------------|
| <span class='code-span'>(\d{3}-\d{2}-\d{4})</span> | Matches a Social Security number format | <span class='code-span'>123-45-6789</span> | <span class='code-span'>123-45-6789</span> |
| <span class='code-span'>(\b\w{4}\b)</span> | Matches any four-letter word    | <span class='code-span'>This is a test</span>      | <span class='code-span'>This</span>, <span class='code-span'>test</span>|
| <span class='code-span'>(?<=\$)\d+</span> | Matches numbers following a <span class='code-span'>$</span> | <span class='code-span'>Cost: $100</span>  | <span class='code-span'>100</span> |
| <span class='code-span'>(abc\|def)</span>  | Matches either <span class='code-span'>abc</span> or <span class='code-span'>def</span>   | <span class='code-span'>abcdef</span>      | <span class='code-span'>abc</span>, <span class='code-span'>def</span>  |
| <span class='code-span'>(?i)regex</span>  | Case-insensitive match for <span class='code-span'>regex</span>      | <span class='code-span'>Regex is fun!</span>       | <span class='code-span'>Regex</span>       |

<!--s-->

## Regular Expressions

Want to practice or make sure your expression works before deploying it? Use an online regex tester!

Live regular expression practice: https://regex101.com/

<!--s-->

## L.08 | Q.01

Which regular expression would match any word that starts with "con" (e.g. captures "conman" but not "icon")?

<div class = 'col-wrapper'>
<div class='c1' style = 'width: 50%; margin-left: 5%'>

A. <span class='code-span'> con\w+ </span><br><br>
B. <span class='code-span'> \bcon\w+ </span><br><br>
C. <span class='code-span'> \bcon\w{3} </span>

</div>
<div class='c2' style = 'width: 50%;'>
<iframe src = 'https://drc-cs-9a3f6.firebaseapp.com/?label=L.08 | Q.01' width = '100%' height = '100%'></iframe>
</div>
</div>

<!--s-->

## L.08 | Q.02

Which regular expression would match any word that ends with "ing" (e.g. captures "running" but not "ring")?

<div class = 'col-wrapper'>
<div class='c1' style = 'width: 50%; margin-left: 5%'>

A. <span class='code-span'> ing\b </span><br><br>
B. <span class='code-span'> \w+ing\b </span><br><br>
C. <span class='code-span'> \w+ing </span>

</div>
<div class='c2' style = 'width: 50%;'>
<iframe src = 'https://drc-cs-9a3f6.firebaseapp.com/?label=L.08 | Q.02' width = '100%' height = '100%'></iframe>
</div>
</div>

<!--s-->

## Regex & OLAP

Regular expressions can be used in OLAP queries to search for text patterns in your data. This is a simple and easy form of text mining that can be done directly in your database.


<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Snowflake

```sql
SELECT *
FROM my_table
WHERE REGEXP_LIKE(phone_number, '\\(\\d{3}\\)\\d{3}-\\d{4}');
```

</div>
<div class="c2" style = "width: 50%">

### BigQuery

```sql
SELECT *
FROM my_table
WHERE REGEXP_CONTAINS(phone_number, r'\(\d{3}\)\d{3}-\d{4}');
```

</div>
</div>

<!--s-->

<div class="header-slide">

# Word Embeddings

</div>

<!--s-->

## Regex vs. Embed

Regular expressions are great for searching for text patterns, but they are limited to the syntax and structure of the text. Word embeddings, on the other hand, allow us to search for the *meaning* of words in a document.

For example, what if you wanted to search for "the sport David Beckham played" in a document that only mentions "soccer" and "Manchester United"? No regex pattern would match this query, but a word embedding / semantic search would.

<!--s-->

## Embed

Word embeddings are dense vector representations of words that capture semantic information. Word embeddings are essential for many NLP tasks because they allow us to work with words in a continuous and meaningful vector space.

**Traditional embeddings** such as Word2Vec are static and pre-trained on large text corpora.

**Contextual embeddings** such as those used by BERT and GPT are dynamic and trained on large language modeling tasks.

<img src="https://miro.medium.com/v2/resize:fit:2000/format:webp/1*SYiW1MUZul1NvL1kc1RxwQ.png" style="margin: 0 auto; display: block; width: 80%; border-radius: 10px;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Google</span>

<!--s-->

## Embed | Traditional Word Embeddings

Word2Vec is a traditional word embedding model that learns word vectors by predicting the context of a word. Word2Vec has two standard architectures:

- **Continuous Bag of Words (CBOW)**. Predicts a word given its context.
- **Skip-gram**. Predicts the context given a word.

Word2Vec is trained on large text corpora and produces dense word vectors that capture semantic information. The result of Word2Vec is a mapping from words to vectors, where similar words are close together in the vector space.

<img src="https://storage.googleapis.com/slide_assets/word2vec.png" style="margin: 0 auto; display: block; width: 50%; border-radius: 10px;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Braun 2017</span>

<!--s-->

## Embed | Traditional Word Embeddings

Traditional word embeddings are static and pre-trained on large text corpora. Some of the most popular traditional word embeddings include Word2Vec, GloVe, and FastText.

The static embeddings they generated were very useful, but they have limitations. They do not capture the context in which a word appears, and they do not adapt to the specific language of a document. This is where contextual embeddings come in.

<!--s-->

## L.08 | Q.03

Which of the following statements is true?

<div class = 'col-wrapper'>
<div class='c1' style = 'width: 50%; margin-left: 5%'>

A. CBOW models predict the context given a word, Skip-gram models predict a word given its context.<br><br>
B. Skip-gram models predict the context given a word, CBOW models predict a word given its context.<br><br>
</div>
<div class='c2' style = 'width: 50%;'>
<iframe src = 'https://drc-cs-9a3f6.firebaseapp.com/?label=L.08 | Q.03' width = '100%' height = '100%'></iframe>
</div>
</div>

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

## Embed | LLM Overview

Transformers are a type of neural network architecture that has been the foundation of NLP advances in recent years. Transformers are powerful because they can capture long-range dependencies in text and are highly parallelizable. You can use embeddings from any of these architectures as "contextual embeddings."

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; font-size: 0.8em;">

**Encoder-Decoder Models**: T5, BART.

Encoder-decoder models generate text by encoding the input text into a fixed-size vector and then decoding the vector into text. Used in machine translation and text summarization.

**Encoder-Only**: BERT

Encoder-only models encode the input text into a fixed-size vector. These models are powerful for text classification tasks but are not typically used for text generation.

**Decoder-Only**: GPT-4, GPT-3, Gemini 

Autoregressive models generate text one token at a time by conditioning on the previous tokens. Used in text generation, language modeling, and summarization.

</div>
<div class="c2 col-centered" style = "width: 50%">

<div>
<img src="https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F81c2aa73-dd8c-46bf-85b0-90e01145b0ed_1422x1460.png" style="margin: 0; padding: 0; ">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Vaswani, 2017</span>
</div>
</div>
</div>

<!--s-->

## L.08 | Q.04

Which of the following architectures provides a word embedding as the **output**?

<div class = 'col-wrapper'>
<div class='c1' style = 'width: 50%; margin-left: 5%'>

A. Encoder-Decoder<br><br>
B. Encoder-Only<br><br>
C. Decoder-Only<br><br>

</div>
<div class='c2' style = 'width: 50%;'>
<iframe src = 'https://drc-cs-9a3f6.firebaseapp.com/?label=L.08 | Q.04' width = '100%' height = '100%'></iframe>
</div>
</div>

<!--s-->

<div class="header-slide">

# Local Semantic Search Demo

<div style = "margin-left: 15%; margin-right: 15%">
Before showing you how this works in Snowflake, let's get a firm understanding of how this works in Python.
</div>

</div>

<!--s-->

## OLAP Semantic Search | Embed

Similar to the local example, we can use Snowflake to embed text data. Snowflake's <span class="code-span">SNOWFLAKE.CORTEX.EMBED_TEXT_768</span> model will take a string of text and return a 768-dimensional vector representation of the text.

```sql
-- Create embedding vectors for issues.
ALTER TABLE issues ADD COLUMN issue_vec VECTOR(FLOAT, 768);

UPDATE issues
  SET issue_vec = SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m', issue_text);
```

<!--s-->

## OLAP Semantic Search | Embed

Here is an example of how you can use Snowflake to embed a query and then perform a semantic search to find the most relevant wiki article for the query. We'll use this technique later in the course when we build a RAG model / chatbot.

```sql

-- Create embedding vectors for issues.
ALTER TABLE issues ADD COLUMN issue_vec VECTOR(FLOAT, 768);

UPDATE issues
  SET issue_vec = SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m', issue_text);

-- Create embedding vector for query.
SELECT
  issue,
  VECTOR_COSINE_SIMILARITY(
    issue_vec,
    SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m', 'User could not install Facebook app on his phone')
  ) AS similarity
FROM issues
ORDER BY similarity DESC
LIMIT 5
WHERE DATEDIFF(day, CURRENT_DATE(), issue_date) < 90 AND similarity > 0.7;

```

<!--s-->

<div class="header-slide">

# Embedding Visualization

</div>

<!--s-->

## Visualization

Let's say you retrieve the word embeddings for a set of words. How can you visualize these embeddings in a way that captures the semantic relationships between the words?

There are two popular approaches to plotting high-dimensional embeddings in 2D space:

- **t-SNE**. t-Distributed Stochastic Neighbor Embedding is a dimensionality reduction technique that captures the local structure of the data.

- **PCA**. Principal Component Analysis is a linear dimensionality reduction technique that captures the global structure of the data.

We'll talk more about PCA later, but for now, let's focus on t-SNE.

<!--s-->

## t-SNE + Visualization

t-SNE is a powerful technique for visualizing high-dimensional data in 2D space. t-SNE works by modeling the similarity between data points in high-dimensional space and then projecting them into 2D space while preserving the local structure of the data.

Here is an example of how you can use t-SNE to visualize word embeddings in Python:

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

```python
from sklearn.manifold import TSNE
import plotly.express as px

# Generate word embeddings.
# ---

# Fit t-SNE model.
tsne = TSNE(n_components=2, random_state=0, perplexity=30)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot embeddings.
px.scatter(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1])
```

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://miro.medium.com/v2/resize:fit:4800/format:webp/0*OFEG6lda-nRTLVYs' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Smetanin 2018</p>
</div>

</div>
</div>

<!--s-->

## t-SNE | Results

<div style='text-align: center;'>
   <img src='https://miro.medium.com/v2/resize:fit:4800/format:webp/0*quUwP6EqEhFmNAqX' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Smetanin 2018</p>
</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Midterm Feedback

  How are things going in MBAI 417? Please provide any feedback or suggestions for improvement, and I'll do my best to accommodate for future lectures. 

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Midterm Feedback" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident are you with **Text Mining** concepts such as:

  - Regular Expressions
  - Semantic Search w/ Embeddings
  - Visualizing Embeddings

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->