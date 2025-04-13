import { NextResponse } from 'next/server';

interface SamplingRequest {
  prompt: string;
  systemPrompt?: string;
  modelName?: string;
  includeContext?: string;
  maxTokens?: number;
  temperature?: number;
}

// Server-side API route to sample from LLM using the backend MCP integration
export async function POST(request: Request) {
  try {
    const body = await request.json() as SamplingRequest;
    
    if (!body.prompt) {
      return NextResponse.json(
        { error: 'prompt is required' },
        { status: 400 }
      );
    }
    
    // Forward the request to the backend API
    const response = await fetch('http://localhost:9997/api/sample', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt: body.prompt,
        system_prompt: body.systemPrompt,
        model_name: body.modelName,
        include_context: body.includeContext || 'thisServer',
        max_tokens: body.maxTokens || 1000,
        temperature: body.temperature
      }),
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      return NextResponse.json(
        { error: errorData.error || 'Failed to sample from LLM' },
        { status: response.status }
      );
    }
    
    const data = await response.json();
    return NextResponse.json(data);
    
  } catch (error) {
    console.error('Error sampling from LLM:', error);
    return NextResponse.json(
      { error: 'Failed to sample from LLM' },
      { status: 500 }
    );
  }
}
