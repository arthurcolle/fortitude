import React from 'react';
import ModelViewer from '../../components/ModelViewer';

export default async function ModelPage({ params }: { params: { model: string } }) {
  const modelName = params.model;
  
  // Fetch data from API server
  const data = await fetchModelData(modelName);
  
  return (
    <div>
      <h2>{modelName}</h2>
      <ModelViewer modelName={modelName} data={data} />
    </div>
  );
}

async function fetchModelData(modelName: string) {
  try {
    // Fetch data from backend API server
    const res = await fetch(`http://localhost:9997/api/${modelName.toLowerCase()}`);
    
    if (!res.ok) {
      return [];
    }
    
    return await res.json();
  } catch (error) {
    console.error('Failed to fetch model data:', error);
    return [];
  }
}
