import React from 'react';

type ModelViewerProps = {
  modelName: string;
  data: any[];
};

export default function ModelViewer({ modelName, data }: ModelViewerProps) {
  if (!data || data.length === 0) {
    return (
      <div>
        <h3>{modelName}</h3>
        <p>No data available</p>
      </div>
    );
  }

  // Get all unique keys from all objects
  const allKeys = Array.from(
    new Set(
      data.flatMap(item => Object.keys(item))
    )
  );

  return (
    <div>
      <h3>{modelName}</h3>
      <table>
        <thead>
          <tr>
            {allKeys.map(key => (
              <th key={key}>{key}</th>
            ))}
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {data.map((item, index) => (
            <tr key={item.id || index}>
              {allKeys.map(key => (
                <td key={key}>{String(item[key] || '')}</td>
              ))}
              <td>
                <button>Edit</button>
                <button>Delete</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
