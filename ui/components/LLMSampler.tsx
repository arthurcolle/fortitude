import React, { useState } from 'react';
import Button from './ui/Button';

type LLMSamplerProps = {
  defaultPrompt?: string;
  defaultSystemPrompt?: string;
  onResult?: (result: string) => void;
};

export default function LLMSampler({ 
  defaultPrompt = '', 
  defaultSystemPrompt = '',
  onResult 
}: LLMSamplerProps) {
  const [prompt, setPrompt] = useState(defaultPrompt);
  const [systemPrompt, setSystemPrompt] = useState(defaultSystemPrompt);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!prompt.trim()) {
      setError('Prompt is required');
      return;
    }
    
    setLoading(true);
    setError('');
    
    try {
      const response = await fetch('/api/sample', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt,
          systemPrompt: systemPrompt || undefined,
          includeContext: 'thisServer',
          maxTokens: 1000,
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to sample from LLM');
      }
      
      const data = await response.json();
      setResult(data.result);
      
      if (onResult) {
        onResult(data.result);
      }
      
    } catch (error) {
      console.error('Error:', error);
      setError(error instanceof Error ? error.message : 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="llm-sampler">
      <h3>LLM Sampler</h3>
      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="system-prompt">System Prompt</label>
          <textarea
            id="system-prompt"
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            rows={2}
            placeholder="Optional system prompt"
          />
        </div>
        
        <div>
          <label htmlFor="prompt">Prompt</label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={4}
            placeholder="Enter your prompt"
            required
          />
        </div>
        
        <Button type="submit" disabled={loading}>
          {loading ? 'Sampling...' : 'Sample'}
        </Button>
      </form>
      
      {error && <div className="error">{error}</div>}
      
      {result && (
        <div className="result">
          <h4>Result:</h4>
          <div className="result-content">{result}</div>
        </div>
      )}
    </div>
  );
}
