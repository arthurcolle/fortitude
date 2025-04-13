import React from 'react';
import { Metadata } from 'next';
import Navigation from '../components/Navigation';

export const metadata: Metadata = {
  title: 'Fortitude Framework',
  description: 'Server-side components with Pydantic models',
};

async function getModels() {
  // In a real app, you'd fetch this from the API server
  // This is a placeholder implementation
  return ['User', 'Product', 'Order'];
}

export default async function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const models = await getModels();
  
  return (
    <html lang="en">
      <body>
        <header>
          <h1>Fortitude Framework</h1>
          <Navigation models={models} />
        </header>
        <main>{children}</main>
        <footer>
          <p>© {new Date().getFullYear()} Fortitude</p>
        </footer>
      </body>
    </html>
  );
}
