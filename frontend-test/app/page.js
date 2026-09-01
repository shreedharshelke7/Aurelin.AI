"use client";

import { useState } from "react";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const askAurelin = async () => {
    alert("Ask button clicked");
    if (!question.trim()) return;
    
    setLoading(true);
    setResult(null);
    setError("");
    

    try {
      const response = await fetch("http://172.20.10.2:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          input_query: question,
        }),
      });

      const data = await response.json();

      if (!data.success) {
        setError(data.error || "Backend request failed.");
        return;
      }

      setResult(data);
    } catch (err) {
      setError("Could not connect to Aurelin backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main style={{ maxWidth: "900px", margin: "40px auto", padding: "20px" }}>
      <h1>Aurelin Frontend Test</h1>

      <div style={{ display: "flex", gap: "10px", marginTop: "30px" }}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a mathematics question..."
          style={{
            flex: 1,
            padding: "12px",
            fontSize: "16px",
          }}
        />

        <button
          onClick={askAurelin}
          disabled={loading}
          style={{
            padding: "12px 20px",
            cursor: "pointer",
          }}
        >
          {loading ? "Generating..." : "Ask"}
        </button>
      </div>

      {loading && (
        <p style={{ marginTop: "20px" }}>
          Aurelin is generating the explanation...
        </p>
      )}

      {error && (
        <p style={{ marginTop: "20px", color: "red" }}>
          {error}
        </p>
      )}

      {result && (
        <div style={{ marginTop: "30px" }}>
          <h2>Narration</h2>
          <p>{result.narration}</p>

          {result.video_url && (
            <>
              <h2>Visual Explanation</h2>

              <video
                src={result.video_url}
                controls
                autoPlay
                style={{
                  width: "100%",
                  marginTop: "10px",
                }}
              />
            </>
          )}
        </div>
      )}
    </main>
  );
}