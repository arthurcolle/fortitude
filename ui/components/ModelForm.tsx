import React, { useState } from 'react';

type FieldDefinition = {
  name: string;
  type: string;
  required: boolean;
  default?: any;
};

type ModelFormProps = {
  modelName: string;
  fields: FieldDefinition[];
  onSubmit: (data: any) => void;
  initialData?: any;
};

export default function ModelForm({ modelName, fields, onSubmit, initialData }: ModelFormProps) {
  const [formData, setFormData] = useState(initialData || {});

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type } = e.target;
    setFormData({
      ...formData,
      [name]: type === 'number' ? Number(value) : value,
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <div>
      <h3>{initialData ? `Edit ${modelName}` : `Create ${modelName}`}</h3>
      <form onSubmit={handleSubmit}>
        {fields.map(field => (
          <div key={field.name}>
            <label htmlFor={field.name}>
              {field.name} {field.required && '*'}
            </label>
            <input
              id={field.name}
              name={field.name}
              type={field.type === 'number' ? 'number' : 'text'}
              value={formData[field.name] || ''}
              onChange={handleChange}
              required={field.required}
            />
          </div>
        ))}
        <button type="submit">{initialData ? 'Update' : 'Create'}</button>
      </form>
    </div>
  );
}
