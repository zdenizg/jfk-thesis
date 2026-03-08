import os
import duckdb
import json
import glob
from flask import Flask, request, jsonify, send_file
from pathlib import Path
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
DB_PATH = "/Users/furkandemir/Desktop/Thesis/database/jfk_files.db"
PDF_ROOT = "/Users/furkandemir/Desktop/Thesis/pdf_files"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not found in environment or .env file.")
    print("Please set it to use the AI chat features.")
MODEL = "llama-3.3-70b-versatile"

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

def get_db_connection():
    return duckdb.connect(DB_PATH, read_only=True)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400

    if not client:
        return jsonify({"error": "GROQ_API_KEY not configured"}), 500

    conn = get_db_connection()
    try:
        # Step 1: Analyze query intent and extract keywords
        analysis_prompt = f"""Analyze this user query for a RAG system searching JFK files: '{query}'
        
        Return a valid JSON object with two keys:
        1. "keywords": A list of 1-3 most important search terms to find relevant documents.
        2. "type": "simple" (if asking for a specific name, date, fact, definition, or single document verification) or "research" (if asking for analysis, summary, relationships, or details on a broad topic).
        
        Reply ONLY with the JSON."""
        
        try:
            analysis_res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            analysis_data = json.loads(analysis_res.choices[0].message.content)
            search_terms = analysis_data.get('keywords', query.split())
            query_type = analysis_data.get('type', 'research')
        except Exception as e:
            print(f"Query keyword analysis failed: {e}")
            search_terms = query.split()
            query_type = 'research'
            
        # Configure strategy based on query type
        if query_type == "simple":
            context_limit = 5
            print(f"Query type: SIMPLE (Terms: {search_terms})")
        else:
            context_limit = 15
            print(f"Query type: RESEARCH (Terms: {search_terms})")
        
        # Build a more flexible search query with basic ranking
        # We rank by the number of keywords matched and total content length (density)
        # to avoid returning empty templates or headers.
        where_clauses = [f"content ILIKE ?" for _ in search_terms]
        ranking_clauses = [f"(CASE WHEN content ILIKE ? THEN 1 ELSE 0 END)" for _ in search_terms]
        
        search_query = f"""
            SELECT DISTINCT content, filename, page_number 
            FROM jfk_pages 
            WHERE ({' OR '.join(where_clauses)}) 
            ORDER BY ({' + '.join(ranking_clauses)}) DESC, length(content) DESC 
            LIMIT 30
        """
        # We need the params twice: once for WHERE and once for ORDER BY ranking
        search_params = [f"%{term}%" for term in search_terms]
        ranking_params = [f"%{term}%" for term in search_terms]
        full_params = search_params + ranking_params
        
        # Step 2: Get global stats to allow AI to answer meta-questions
        total_p = conn.execute("SELECT COUNT(*) FROM jfk_pages").fetchone()[0]
        hw_p = conn.execute("SELECT COUNT(*) FROM jfk_pages WHERE includes_handwriting = true").fetchone()[0]
        stamp_p = conn.execute("SELECT COUNT(*) FROM jfk_pages WHERE has_stamps = true").fetchone()[0]
        redact_p = conn.execute("SELECT COUNT(*) FROM jfk_pages WHERE has_redactions = true").fetchone()[0]
        
        results = conn.execute(search_query, full_params).fetchall()
        
        # Deduplicate and clean results
        unique_results = []
        seen_content = set()
        for r in results:
            content_snippet = r[0][:200].strip() # Check first 200 chars for duplicates
            if content_snippet not in seen_content:
                unique_results.append(r)
                seen_content.add(content_snippet)
        
        # Limit to unique segments based on query complexity
        final_results = unique_results[:context_limit]
        
        context = ""
        if final_results:
            context = "\n\n".join([f"Source: {r[1]} Page: {r[2]}\nContent: {r[0]}" for r in final_results])
        else:
            context = "NO UNIQUE SEARCH RESULTS FOUND FOR CONTENT."

        # Construct dynamic prompt instructions
        if query_type == "simple":
            instructions = """
            You are a helpful research assistant. 
            Provide a direct, concise answer to the user's question using ONLY the provided context.
            - If the answer is a specific name, date, or fact, state it clearly.
            - Mention the specific source document (filename) and page number for the fact found.
            - Do NOT provide a full research report. Keep it brief and to the point.
            """
        else:
            instructions = """
            You are JFK Files GPT, a senior Research Historian developed for the Master of Statistics thesis at KU Leuven. 
            Your goal is to provide a comprehensive, highly structured Research Report based on retrieved archival documents.

            REPORTING RULES:
            1. USE MARKDOWN: Use headers (##), bold text, bullet points, and tables to organize information.
            2. BE COMPREHENSIVE: Do not summarize briefly. Extract every relevant detail found in the context.
            3. CATEGORIZE: Group findings into logical sections (e.g., "Background", "Specific Activities", "Internal Agency Communcations").
            4. SOURCE CITE: Every section should clearly mention which PDF and page it came from.
            5. STRICTNESS: Use ONLY the provided Context and Global Stats. No outside information.
            6. NO REPETITION: Do not repeat facts.
            """

        prompt = f"""{instructions}

        GLOBAL DATABASE CONTEXT:
        - Total Archive Size: {total_p} pages
        - Detected Handwriting: {hw_p} pages
        - Official Stamps: {stamp_p} pages
        - Redacted Content: {redact_p} pages

        Context:
        {context}
        
        User Inquiry: {query}
        """

        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        return jsonify({
            "answer": completion.choices[0].message.content,
            "sources": [{"filename": r[1], "page": r[2]} for r in final_results],
            "query_type": query_type
        })
    except Exception as e:
        print(f"Error in /api/chat: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/stats', methods=['GET'])
def stats():
    conn = get_db_connection()
    try:
        # Basic counts
        total_pages = conn.execute("SELECT COUNT(*) FROM jfk_pages").fetchone()[0]
        total_docs = conn.execute("SELECT COUNT(DISTINCT file_id) FROM jfk_pages").fetchone()[0]
        
        # Content-based counts
        pages_with_content = conn.execute("SELECT COUNT(*) FROM jfk_pages WHERE content IS NOT NULL AND length(trim(content)) > 0").fetchone()[0]
        docs_with_content = conn.execute("SELECT COUNT(DISTINCT file_id) FROM jfk_pages WHERE content IS NOT NULL AND length(trim(content)) > 0").fetchone()[0]

        # Percentages
        page_pct = (pages_with_content / total_pages * 100) if total_pages > 0 else 0
        doc_pct = (docs_with_content / total_docs * 100) if total_docs > 0 else 0

        # Metadata counts
        handwritten_pages = conn.execute("SELECT COUNT(*) FROM jfk_pages WHERE includes_handwriting = true").fetchone()[0]
        stamped_pages = conn.execute("SELECT COUNT(*) FROM jfk_pages WHERE has_stamps = true").fetchone()[0]
        redacted_pages = conn.execute("SELECT COUNT(*) FROM jfk_pages WHERE has_redactions = true").fetchone()[0]

        doc_types = conn.execute("SELECT document_type, COUNT(*) as count FROM jfk_pages GROUP BY document_type ORDER BY count DESC LIMIT 5").fetchall()

        return jsonify({
            "total_pages": total_pages,
            "total_docs": total_docs,
            "pages_with_content": pages_with_content,
            "docs_with_content": docs_with_content,
            "page_content_pct": round(page_pct, 1),
            "doc_content_pct": round(doc_pct, 1),
            "handwritten_pages": handwritten_pages,
            "stamped_pages": stamped_pages,
            "redacted_pages": redacted_pages,
            "document_types": [{"type": r[0], "count": r[1]} for r in doc_types]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    action = data.get('action') # 'names', 'summarize'
    text = data.get('text')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
        
    if not client:
        return jsonify({"error": "GROQ_API_KEY not configured"}), 500

    if action == 'names':
        prompt = f"Extract all unique people's names from the following text. Return them as a comma-separated list. If none, say 'None'.\n\nText: {text}"
    elif action == 'summarize':
        prompt = f"Provide a concise summary of the following text, highlighting key information and events.\n\nText: {text}"
    else:
        return jsonify({"error": "Invalid action"}), 400

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return jsonify({"result": completion.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/pdf/<filename>', methods=['GET'])
def get_pdf(filename):
    # Search for the file in the nested structure
    # Using glob for recursive search
    search_pattern = os.path.join(PDF_ROOT, "**", filename)
    files = glob.glob(search_pattern, recursive=True)
    
    if files:
        # Return the first match found
        return send_file(files[0], mimetype='application/pdf')
    else:
        return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
