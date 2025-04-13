import { NextResponse } from 'next/server';

// Server-side API route to get available models
export async function GET() {
  try {
    // In a real app, you would fetch this from the backend API
    // This is a placeholder implementation
    const models = ['User', 'Product', 'Order'];
    
    return NextResponse.json({ models });
  } catch (error) {
    console.error('Error fetching models:', error);
    return NextResponse.json(
      { error: 'Failed to fetch models' },
      { status: 500 }
    );
  }
}
