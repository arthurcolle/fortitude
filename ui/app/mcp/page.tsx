import React from 'react';
import LLMSampler from '../../components/LLMSampler';

export default function MCPPage() {
  return (
    <div>
      <h2>MCP Sampling</h2>
      <p>
        This page demonstrates the MCP (Model Context Protocol) integration in Fortitude.
        You can use the sampler below to request completions from LLMs through the MCP.
      </p>
      
      <LLMSampler 
        defaultPrompt="Explain how Model Context Protocol works in Fortitude"
        defaultSystemPrompt="You are a helpful assistant that explains technical concepts clearly and concisely."
      />
      
      <div className="info-box">
        <h3>How MCP Works in Fortitude</h3>
        <p>
          When you submit a prompt through this interface:
        </p>
        <ol>
          <li>The UI sends your request to the Next.js API route</li>
          <li>The API route forwards it to the Fortitude backend server</li>
          <li>The backend uses the MCP client to send a sampling request to the MCP-enabled client</li>
          <li>The client shows the user the prompt and requests approval</li>
          <li>After approval, the client samples from an LLM</li>
          <li>The result is sent back through the same path to this UI</li>
        </ol>
        <p>
          This human-in-the-loop design ensures users maintain control over what the LLM sees and generates.
        </p>
      </div>
    </div>
  );
}