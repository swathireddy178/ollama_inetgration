# ollama_inetgration
# RAG Chatbot using LangChain, Ollama and Gemma 2B

## Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot that answers questions based on website content.

The application loads data from a website, splits the content into chunks, creates embeddings, stores them in a vector database, retrieves relevant information, and uses the Gemma 2B model running locally through Ollama to generate answers.

## Features

- Website Data Ingestion
- Text Chunking
- Embeddings Generation
- Semantic Search
- Vector Database Storage
- Retrieval-Augmented Generation (RAG)
- Local LLM using Ollama
- Gemma 2B Integration
- Streamlit User Interface

## Tech Stack

- Python
- LangChain
- Ollama
- Gemma 2B
- ChromaDB / FAISS
- Streamlit

## Project Architecture

Website
→ WebBaseLoader
→ Text Splitter
→ Embeddings
→ Vector Database
→ Retriever
→ Gemma 2B (Ollama)
→ Final Answer

## Installation

```bash
pip install -r requirements.txt
