import React from 'react';
import ModelForm from '../../../components/ModelForm';

export default async function NewModelPage({ params }: { params: { model: string } }) {
  const modelName = params.model;
  
  // Fetch model schema from API server
  const fields = await fetchModelSchema(modelName);
  
  return (
    <div>
      <h2>Create New {modelName}</h2>
      <ModelForm 
        modelName={modelName} 
        fields={fields}
        onSubmit={async (data) => {
          'use server';
          // Server action to submit the form
          await createModel(modelName, data);
        }} 
      />
    </div>
  );
}

async function fetchModelSchema(modelName: string) {
  try {
    // In production, you would fetch this from your API server
    // This is a placeholder implementation
    return [
      { name: 'name', type: 'string', required: true },
      { name: 'description', type: 'string', required: false },
    ];
  } catch (error) {
    console.error('Failed to fetch model schema:', error);
    return [];
  }
}

async function createModel(modelName: string, data: any) {
  try {
    const response = await fetch(`http://localhost:9997/api/${modelName.toLowerCase()}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
    
    if (!response.ok) {
      throw new Error(`Failed to create ${modelName}`);      
    }
    
    return await response.json();
  } catch (error) {
    console.error(`Error creating ${modelName}:`, error);
    throw error;
  }
}
