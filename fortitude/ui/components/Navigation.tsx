import React from 'react';
import Link from 'next/link';

type NavigationProps = {
  models: string[];
};

export default function Navigation({ models }: NavigationProps) {
  return (
    <nav>
      <ul>
        <li>
          <Link href="/">Home</Link>
        </li>
        {models.map(model => (
          <li key={model}>
            <Link href={`/${model}`}>{model}</Link>
            <Link href={`/new/${model}`}>(+)</Link>
          </li>
        ))}
      </ul>
    </nav>
  );
}
