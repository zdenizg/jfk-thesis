import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Send, PieChart, Users, FileText, Search, Activity, Sparkles } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const API_BASE = 'http://localhost:5001/api';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    fetchStats();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const fetchStats = async () => {
    try {
      const res = await axios.get(`${API_BASE}/stats`);
      setStats(res.data);
    } catch (err) {
      console.error("Error fetching stats:", err);
    }
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const res = await axios.post(`${API_BASE}/chat`, { query: input });
      const aiMsg = {
        role: 'ai',
        content: res.data.answer,
        sources: res.data.sources
      };
      setMessages(prev => [...prev, aiMsg]);
    } catch (err) {
      const errorMsg = err.response?.data?.error || "Sorry, I encountered an error processing your request.";
      setMessages(prev => [...prev, { role: 'ai', content: errorMsg }]);
    } finally {
      setLoading(false);
    }
  };

  const analyzeContent = async (text, action) => {
    try {
      setLoading(true);
      const res = await axios.post(`${API_BASE}/analyze`, { text, action });
      const aiMsg = {
        role: 'ai',
        content: `### ${action.toUpperCase()} ANALYSIS\n\n${res.data.result}`
      };
      setMessages(prev => [...prev, aiMsg]);
    } catch (err) {
      console.error("Analysis error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="sidebar">
        <div className="logo">JFK Files GPT</div>

        <div className="thesis-info" style={{ marginBottom: '2rem', fontSize: '0.8rem', color: 'var(--text-muted)', borderLeft: '2px solid var(--primary)', paddingLeft: '1rem' }}>
          <p style={{ fontWeight: '600', color: 'var(--text-main)', marginBottom: '0.25rem' }}>Master of Statistics and Data Science</p>
          <p style={{ marginBottom: '0.5rem' }}>KU Leuven</p>
          <p style={{ fontStyle: 'italic' }}>Thesis: "Topic Modeling and Thematic Analysis of JFK Assassination Files Using NLP"</p>
        </div>

        <div className="stats-section">
          <h3 style={{ marginBottom: '1rem', color: 'var(--text-muted)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Database Overview</h3>
          <div style={{ display: 'grid', gap: '1rem' }}>
            <div className="stat-card">
              <span className="stat-value">{stats?.total_docs.toLocaleString() || '...'}</span>
              <span className="stat-label">Total Documents</span>
            </div>
            <div className="stat-card">
              <span className="stat-value">{stats?.total_pages.toLocaleString() || '...'}</span>
              <span className="stat-label">Total Pages</span>
            </div>

            <div style={{ marginTop: '0.5rem', display: 'grid', gap: '0.5rem' }}>
              <div className="mini-stat">
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span className="stat-label">Docs w/ Content</span>
                  <span style={{ color: 'var(--primary)' }}>{stats?.doc_content_pct}%</span>
                </div>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${stats?.doc_content_pct}%` }}></div>
                </div>
              </div>

              <div className="mini-stat">
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span className="stat-label">Pages w/ Content</span>
                  <span style={{ color: 'var(--primary)' }}>{stats?.page_content_pct}%</span>
                </div>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${stats?.page_content_pct}%` }}></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="tools-section" style={{ marginTop: 'auto' }}>
          <h3 style={{ marginBottom: '1rem', color: 'var(--text-muted)', fontSize: '0.75rem', textTransform: 'uppercase' }}>Analysis Tools</h3>
          <div style={{ display: 'grid', gap: '0.5rem' }}>
            <button className="tool-btn" onClick={() => analyzeContent(messages[messages.length - 1]?.content, 'names')} disabled={messages.length === 0}>
              <Users size={18} /> Look for Names
            </button>
            <button className="tool-btn" onClick={() => analyzeContent(messages[messages.length - 1]?.content, 'summarize')} disabled={messages.length === 0}>
              <FileText size={18} /> Summarize Context
            </button>
          </div>
        </div>
      </div>

      <div className="main-content">
        <div className="chat-history">
          {messages.length === 0 && (
            <div style={{ margin: 'auto', textAlign: 'center', opacity: 0.5 }}>
              <Sparkles size={48} style={{ marginBottom: '1rem' }} />
              <h2>How can I help you explore the JFK Archives?</h2>
              <p>Ask about specific documents, people, or events.</p>
            </div>
          )}
          <AnimatePresence>
            {messages.map((msg, i) => (
              <motion.div
                key={i}
                className={`message ${msg.role}`}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <div className="msg-content">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {msg.content}
                  </ReactMarkdown>
                </div>
                {msg.sources && msg.sources.length > 0 && (
                  <div className="sources">
                    {msg.sources.map((s, si) => (
                      <a
                        key={si}
                        href={`${API_BASE}/pdf/${s.filename}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="source-tag"
                      >
                        {s.filename} (p. {s.page})
                      </a>
                    ))}
                  </div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>
          {loading && (
            <div className="message ai" style={{ opacity: 0.7 }}>
              <div className="typing-indicator">Analyzing document database...</div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className="input-container">
          <form className="input-wrapper" onSubmit={handleSend}>
            <Search size={20} color="var(--text-muted)" />
            <input
              type="text"
              placeholder="Search JFK documents..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
            <button type="submit" className="send-btn">
              <Send size={18} />
            </button>
          </form>
        </div>
      </div>

      <style jsx>{`
        .tool-btn {
          background: var(--glass);
          border: 1px solid var(--border);
          color: white;
          padding: 0.75rem 1rem;
          border-radius: 0.75rem;
          display: flex;
          align-items: center;
          gap: 0.75rem;
          cursor: pointer;
          transition: all 0.2s;
          font-size: 0.875rem;
        }
        .tool-btn:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.1);
          border-color: var(--primary);
        }
        .tool-btn:disabled {
          opacity: 0.3;
          cursor: not-allowed;
        }
        .typing-indicator {
          display: flex;
          gap: 4px;
        }
        .msg-content h2, .msg-content h3 {
          margin: 1rem 0 0.5rem 0;
          color: var(--primary);
        }
        .msg-content ul, .msg-content ol {
          margin-left: 1.5rem;
          margin-bottom: 1rem;
        }
        .msg-content p {
          margin-bottom: 0.5rem;
        }
      `}</style>
    </div>
  );
}

export default App;
