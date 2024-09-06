# RAG Techniques Study

This project explores various advanced Retrieval-Augmented Generation (RAG) techniques to enhance information retrieval and question-answering systems.

## Terminology

- RAG: Retrieval-Augmented Generation
- LLM: Large Language Model
- Embedding: Vector representation of text
- Vectorstore: Database for storing and querying vector embeddings
- Chunking: Splitting text into smaller, manageable pieces
- Retriever: Component that fetches relevant information from a knowledge base

## Techniques Explored

1. Simple RAG: Basic implementation using FAISS vectorstore and OpenAI embeddings.
2. Adaptive Retrieval: Dynamically adjusts retrieval strategy based on query type.
3. CRAG (Corrective RAG): Implements a dynamic correction process for retrieved information.
4. Self-RAG: Combines retrieval and generation with self-evaluation capabilities.
5. GraphRAG: Utilizes graph-based knowledge representation for enhanced retrieval.
6. RAPTOR: Implements Recursive Abstractive Processing and Thematic Organization for Retrieval.
7. Hierarchical Indices: Creates a multi-level structure of document summaries for efficient retrieval.
8. Semantic Chunking: Uses context-aware text segmentation for more meaningful retrieval.
9. Reliable RAG: Incorporates validation steps to ensure accuracy and relevance of retrieved information.

## Key Components

- Document loading and preprocessing
- Text chunking and cleaning
- Embedding generation
- Vectorstore creation (often using FAISS)
- Query processing and classification
- Retrieval strategies
- Answer generation and refinement

## New Methods and Approaches

### 1. Query classification for adaptive retrieval:

```
"from langchain.embeddings import OpenAIEmbeddings\n",
"from langchain.text_splitter import CharacterTextSplitter\n",
"from langchain.prompts import PromptTemplate\n",
```

### 2. Hierarchical summarization for document processing:

```
"    summary_chain = load_summarize_chain(summary_llm, chain_type=\"map_reduce\")\n",
"    \n",
"    async def summarize_doc(doc):\n",
"        \"\"\"\n",
"        Summarizes a single document with rate limit handling.\n",
"        \n",
"        Args:\n",
"            doc: The document to be summarized.\n",
"            \n",
"        Returns:\n",
"            A summarized Document object.\n",
"        \"\"\"\n",
"        # Retry the summarization with exponential backoff\n",
"        summary_output = await retry_with_exponential_backoff(summary_chain.ainvoke([doc]))\n",
"        summary = summary_output['output_text']\n",
"        return Document(\n",
"            page_content=summary,\n",
"            metadata={\"source\": path, \"page\": doc.metadata[\"page\"], \"summary\": True}\n",
"        )\n",
"\n",
"    # Process documents in smaller batches to avoid rate limits\n",
"    batch_size = 5  # Adjust this based on your rate limits\n",
"    summaries = []\n",
"    for i in range(0, len(documents), batch_size):\n",
"        batch = documents[i:i+batch_size]\n",
"        batch_summaries = await asyncio.gather(*[summarize_doc(doc) for doc in batch])\n",
"        summaries.extend(batch_summaries)\n",
"        await asyncio.sleep(1)  # Short pause between batches\n",
"\n",
"    # Split documents into detailed chunks\n",
"    text_splitter = RecursiveCharacterTextSplitter(\n",
"        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len\n",
"    )\n",
"    detailed_chunks = await asyncio.to_thread(text_splitter.split_documents, documents)\n",
"\n",
"    # Update metadata for detailed chunks\n",
"    for i, chunk in enumerate(detailed_chunks):\n",
"        chunk.metadata.update({\n",
"            \"chunk_id\": i,\n",
"            \"summary\": False,\n",
"            \"page\": int(chunk.metadata.get(\"page\", 0))\n",
"        })\n",
"\n",
"    # Create embeddings\n",
"    embeddings = OpenAIEmbeddings()\n",
"\n",
"    # Create vector stores asynchronously with rate limit handling\n",
```

### 3. Graph-based knowledge representation:

```
"        named_entities = [ent.text for ent in doc.ents if ent.label_ in [\"PERSON\", \"ORG\", \"GPE\", \"WORK_OF_ART\"]]\n",
"        \n",
"        # Extract general concepts using LLM\n",
"        concept_extraction_prompt = PromptTemplate(\n",
"            input_variables=[\"text\"],\n",
"            template=\"Extract key concepts (excluding named entities) from the following text:\\n\\n{text}\\n\\nKey concepts:\"\n",
"        )\n",
"        concept_chain = concept_extraction_prompt | llm.with_structured_output(Concepts)\n",
"        general_concepts = concept_chain.invoke({\"text\": content}).concepts_list\n",
"        \n",
"        # Combine named entities and general concepts\n",
"        all_concepts = list(set(named_entities + general_concepts))\n",
"        \n",
"        self.concept_cache[content] = all_concepts\n",
"        return all_concepts\n",
```

### 4. Self-evaluation and iterative refinement in Self-RAG:

```
"        docs = vectorstore.similarity_search(query, k=top_k)\n",
"        contexts = [doc.page_content for doc in docs]\n",
"        print(f\"Retrieved {len(contexts)} documents\")\n",
"        \n",
"        # Step 3: Evaluate relevance of retrieved documents\n",
"        print(\"Step 3: Evaluating relevance of retrieved documents...\")\n",
"        relevant_contexts = []\n",
"        for i, context in enumerate(contexts):\n",
"            input_data = {\"query\": query, \"context\": context}\n",
"            relevance = relevance_chain.invoke(input_data).response.strip().lower()\n",
"            print(f\"Document {i+1} relevance: {relevance}\")\n",
"            if relevance == 'relevant':\n",
"                relevant_contexts.append(context)\n",
"        \n",
"        print(f\"Number of relevant contexts: {len(relevant_contexts)}\")\n",
"        \n",
"        # If no relevant contexts found, generate without retrieval\n",
"        if not relevant_contexts:\n",
"            print(\"No relevant contexts found. Generating without retrieval...\")\n",
```

### 5. Semantic chunking for context-aware text splitting:

```
"os.environ[\"OPENAI_API_KEY\"] = os.getenv('OPENAI_API_KEY')\n",
"\n"
]
},
{
"cell_type": "markdown",
"metadata": {},
"source": [
"### Define file path"
]
},
{
"cell_type": "code",
"execution_count": 3,
"metadata": {},
"outputs": [],
```

## Benefits of Advanced RAG Techniques

- Improved accuracy and relevance of retrieved information
- Better handling of complex queries and diverse document types
- Enhanced context preservation and understanding
- Increased efficiency in processing large document collections
- More flexible and adaptable information retrieval systems
