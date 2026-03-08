# JFK Files RAG System

A simple but powerful Retrieval-Augmented Generation system for the JFK Archives.

## Features
- **Document Q&A**: Ask questions about the documents using Groq & Llama 3.3.
- **Statistical Analysis**: Real-time stats from the DuckDB database.
- **Entity Extraction**: Look for people's names in retrieved documents.
- **Summarization**: Summarize long document contents.
- **Premium UI**: Modern dark-mode interface with smooth animations.

## Tech Stack
- **Backend**: Python, Flask, DuckDB, Groq.
- **Frontend**: React, Vite, Framer Motion, Lucide Icons.

## Setup
1. Create a `.env` file in this directory and add your `GROQ_API_KEY`:
   ```
   GROQ_API_KEY=gsk_...
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   cd frontend && npm install
   ```
3. Run the application:
   ```bash
   npm run dev
   ```

## Database
Uses the database at `/Users/furkandemir/Desktop/Thesis/database/jfk_files.db`.
