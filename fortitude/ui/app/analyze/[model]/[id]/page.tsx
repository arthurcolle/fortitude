import React from 'react';
import LLMSampler from '../../../../components/LLMSampler';

export default async function AnalyzePage({ params }: { params: { model: string, id: string } }) {
  const { model, id } = params;
  
  // Fetch the model instance
  const instance = await fetchModelInstance(model, id);
  
  if (!instance) {
    return (
      <div>
        <h2>Error</h2>
        <p>Could not find {model} with ID {id}</p>
      </div>
    );
  }
  
  // Format the data for display
  const formattedData = JSON.stringify(instance, null, 2);
  
  // Create default prompt for analysis
  const defaultPrompt = `Analyze this ${model} data and provide insights.`;
  const defaultSystemPrompt = `You are an expert at analyzing ${model} data. Provide concise, accurate insights.`;
  
  return (
    <div>
      <h2>Analyze {model}: {instance.name || instance.id}</h2>
      
      <div className="data-display">
        <h3>Data:</h3>
        <pre>{formattedData}</pre>
      </div>
      
      <LLMSampler 
        defaultPrompt={defaultPrompt} 
        defaultSystemPrompt={defaultSystemPrompt} 
      />
    </div>
  );
}

async function fetchModelInstance(modelName: string, id: string) {
  try {
    // Fetch data from backend API server
    const res = await fetch(`http://localhost:9997/api/${modelName.toLowerCase()}/${id}`);
    
    if (!res.ok) {
      return null;
    }
    
    return await res.json();
  } catch (error) {
    console.error(`Failed to fetch ${modelName} with ID ${id}:`, error);
    return null;
  }
}
