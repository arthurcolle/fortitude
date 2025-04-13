#!/usr/bin/env python3

import os
import sys
import re
import json
import importlib.util
import textwrap
from typing import Dict, List, Any, Optional, Union, Type, Tuple
from pathlib import Path
import shutil

# Template for an advanced model with relationships and validations
ADVANCED_MODEL_TEMPLATE = """from fortitude.backend.models import FortitudeBaseModel
from pydantic import Field, validator, root_validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import uuid

class {model_name}(FortitudeBaseModel):
    """"{docstring}"""
    # Identifying fields
    name: str = Field(..., description="Name of the {model_name}")
    slug: Optional[str] = Field(None, description="URL-friendly slug")
    
    # Main fields
{fields}
    
    # Relationship fields
{relations}
    
    # Metadata
    status: str = Field("active", description="Status of this {model_name}")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    # Computed properties
    @property
    def is_active(self) -> bool:
        """Returns whether this {model_name} is active"""
        return self.status == "active"
    
    # Validators
    @validator("slug", pre=True, always=True)
    def generate_slug(cls, v, values):
        """Generate a slug from the name if not provided"""
        if v is None and "name" in values:
            # Convert name to lowercase, replace spaces with hyphens, remove special chars
            return re.sub(r'[^a-z0-9-]', '', values["name"].lower().replace(" ", "-"))
        return v
    
    # Root validator for complex validations across fields
    @root_validator
    def check_consistency(cls, values):
        """Validate consistency across fields"""
        # Example validation logic
        return values
"""

# Template for an advanced service with caching and transactions
ADVANCED_SERVICE_TEMPLATE = """from typing import Dict, List, Any, Optional, Union, TypeVar, Generic
from pydantic import parse_obj_as
import asyncio
import json
import logging
from datetime import datetime, timedelta
from uuid import uuid4
from ..models.{model_file} import {model_name}
from ..cache import Cache
from ..mcp.client import MCPClient

# Type variable for the model
T = TypeVar('T', bound='{model_name}')

logger = logging.getLogger(__name__)

class {service_name}:
    """Advanced service for {model_name} with caching and transactions"""
    
    def __init__(self):
        # In-memory storage for demo purposes
        # In a real app, this would use a database
        self._data: Dict[str, {model_name}] = {{}}
        
        # Set up caching with 5-minute TTL
        self._cache = Cache(ttl=timedelta(minutes=5))
        
        # MCP client for LLM capabilities
        self.mcp_client = MCPClient()
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure service-specific logging"""
        self._logger = logging.getLogger(f"{service_name}")
        # Add a handler if needed
    
    # ----- CRUD Operations -----
    
    async def create(self, data: Dict[str, Any]) -> {model_name}:
        """Create a new {model_name} with transaction support"""
        # Start transaction (simulated)
        transaction_id = str(uuid4())
        self._logger.info(f"Starting transaction {{transaction_id}} for create operation")
        
        try:
            # Create the instance
            instance = {model_name}(**data)
            
            # Store in-memory (would be a DB insert in a real app)
            self._data[instance.id] = instance
            
            # Invalidate cache
            cache_key = f"all_{model_name_lower}s"
            self._cache.invalidate(cache_key)
            
            # Commit transaction (simulated)
            self._logger.info(f"Committing transaction {{transaction_id}}")
            
            return instance
        except Exception as e:
            # Rollback transaction (simulated)
            self._logger.error(f"Error in transaction {{transaction_id}}: {{str(e)}}")
            self._logger.info(f"Rolling back transaction {{transaction_id}}")
            raise
    
    async def get(self, id: str) -> Optional[{model_name}]:
        """Get a {model_name} by ID with caching"""
        # Check cache first
        cache_key = f"{model_name_lower}_{{id}}"
        cached = self._cache.get(cache_key)
        
        if cached:
            self._logger.debug(f"Cache hit for {{cache_key}}")
            return parse_obj_as({model_name}, json.loads(cached))
        
        # Cache miss, get from storage
        self._logger.debug(f"Cache miss for {{cache_key}}")
        instance = self._data.get(id)
        
        if instance:
            # Update cache
            self._cache.set(cache_key, instance.model_dump_json())
        
        return instance
    
    async def list(self, filters: Dict[str, Any] = None, 
                  sort_by: str = None, 
                  sort_order: str = "asc",
                  page: int = 1,
                  page_size: int = 100) -> List[{model_name}]:
        """List {model_name}s with filtering, sorting, and pagination"""
        # Start with all items
        items = list(self._data.values())
        
        # Apply filters if provided
        if filters:
            for key, value in filters.items():
                items = [item for item in items if getattr(item, key, None) == value]
        
        # Apply sorting
        if sort_by:
            reverse = sort_order.lower() == "desc"
            items.sort(key=lambda x: getattr(x, sort_by, None), reverse=reverse)
        
        # Apply pagination
        start = (page - 1) * page_size
        end = start + page_size
        
        return items[start:end]
    
    async def update(self, id: str, data: Dict[str, Any]) -> Optional[{model_name}]:
        """Update a {model_name} with transaction support"""
        # Check if exists
        if id not in self._data:
            return None
        
        # Start transaction (simulated)
        transaction_id = str(uuid4())
        self._logger.info(f"Starting transaction {{transaction_id}} for update operation")
        
        try:
            # Get existing data and update with new data
            existing_data = self._data[id].model_dump()
            updated_data = {{**existing_data, **data, "id": id, "updated_at": datetime.now()}}
            
            # Create updated instance
            instance = {model_name}(**updated_data)
            
            # Update storage
            self._data[id] = instance
            
            # Invalidate caches
            self._cache.invalidate(f"{model_name_lower}_{{id}}")
            self._cache.invalidate(f"all_{model_name_lower}s")
            
            # Commit transaction (simulated)
            self._logger.info(f"Committing transaction {{transaction_id}}")
            
            return instance
        except Exception as e:
            # Rollback transaction (simulated)
            self._logger.error(f"Error in transaction {{transaction_id}}: {{str(e)}}")
            self._logger.info(f"Rolling back transaction {{transaction_id}}")
            raise
    
    async def delete(self, id: str) -> bool:
        """Delete a {model_name} with transaction support"""
        if id not in self._data:
            return False
        
        # Start transaction (simulated)
        transaction_id = str(uuid4())
        self._logger.info(f"Starting transaction {{transaction_id}} for delete operation")
        
        try:
            # Delete from storage
            del self._data[id]
            
            # Invalidate caches
            self._cache.invalidate(f"{model_name_lower}_{{id}}")
            self._cache.invalidate(f"all_{model_name_lower}s")
            
            # Commit transaction (simulated)
            self._logger.info(f"Committing transaction {{transaction_id}}")
            
            return True
        except Exception as e:
            # Rollback transaction (simulated)
            self._logger.error(f"Error in transaction {{transaction_id}}: {{str(e)}}")
            self._logger.info(f"Rolling back transaction {{transaction_id}}")
            raise
    
    # ----- Advanced Operations -----
    
    async def analyze(self, id: str, question: str) -> str:
        """Analyze a {model_name} with LLM"""
        instance = await self.get(id)
        if not instance:
            return f"Could not find {model_name} with id {{id}}"
        
        prompt = f"Based on this {model_name} data:\\n{{instance.model_dump_json(indent=2)}}\\n\\nQuestion: {{question}}"
        system_prompt = f"You are an expert at analyzing {model_name} data. Provide concise, accurate insights."
        
        try:
            return await self.mcp_client.sample_text(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=1000
            )
        except Exception as e:
            self._logger.error(f"Error analyzing data: {{str(e)}}")
            return f"Error analyzing data: {{str(e)}}"
    
    async def bulk_create(self, items: List[Dict[str, Any]]) -> List[{model_name}]:
        """Create multiple {model_name}s in a single transaction"""
        results = []
        transaction_id = str(uuid4())
        self._logger.info(f"Starting bulk transaction {{transaction_id}} for {{len(items)}} items")
        
        try:
            for item in items:
                instance = await self.create(item)
                results.append(instance)
            
            self._logger.info(f"Committed bulk transaction {{transaction_id}}")
            return results
        except Exception as e:
            self._logger.error(f"Error in bulk transaction {{transaction_id}}: {{str(e)}}")
            self._logger.info(f"Rolling back bulk transaction {{transaction_id}}")
            # In a real app, this would roll back all previous operations
            raise
    
    async def export(self, format: str = "json") -> str:
        """Export all {model_name}s in specified format"""
        items = list(self._data.values())
        
        if format.lower() == "json":
            return json.dumps([item.model_dump() for item in items], default=str, indent=2)
        elif format.lower() == "csv":
            # Simple CSV implementation (would use csv module in real app)
            if not items:
                return "id,name,created_at,updated_at"
                
            headers = list(items[0].model_dump().keys())
            result = ",".join(headers) + "\\n"
            
            for item in items:
                values = [str(item.model_dump().get(h, "")) for h in headers]
                result += ",".join(values) + "\\n"
                
            return result
        else:
            raise ValueError(f"Unsupported export format: {{format}}")
"""

# Template for an advanced controller with auth and advanced features
ADVANCED_CONTROLLER_TEMPLATE = """from typing import Dict, List, Any, Optional, Union, Annotated
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, Header
from fastapi.responses import JSONResponse, StreamingResponse
import io
import json
from datetime import datetime
from ..models.{model_file} import {model_name}
from ..services.{service_file} import {service_name}
from ..auth import get_current_user, User, requires_role

router = APIRouter(prefix="/{route_prefix}", tags=["{tag}"])
service = {service_name}()

# ----- CRUD Operations -----

@router.post("/", response_model={model_name})
async def create_{operation_name}(
    data: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_user)
):
    """Create a new {name}"""
    # Check if user has permission
    requires_role(current_user, ["admin", "editor"])
    
    # Add audit fields
    data["created_by"] = current_user.id
    
    return await service.create(data)

@router.get("/{{{id_param}}}", response_model={model_name})
async def get_{operation_name}(
    id: str = Path(..., description="The ID of the {name} to get"),
    current_user: User = Depends(get_current_user)
):
    """Get a {name} by ID"""
    result = await service.get(id)
    if not result:
        raise HTTPException(status_code=404, detail="{model_name} not found")
    return result

@router.get("/", response_model=List[{model_name}])
async def list_{operation_name}(
    status: Optional[str] = Query(None, description="Filter by status"),
    sort_by: Optional[str] = Query(None, description="Field to sort by"),
    sort_order: Optional[str] = Query("asc", description="Sort order (asc or desc)"),
    page: int = Query(1, description="Page number", ge=1),
    page_size: int = Query(100, description="Items per page", ge=1, le=1000),
    current_user: User = Depends(get_current_user)
):
    """List all {name}s with filtering, sorting, and pagination"""
    # Build filters dict from query params
    filters = {{"status": status}} if status else {{}}
    
    return await service.list(
        filters=filters,
        sort_by=sort_by,
        sort_order=sort_order,
        page=page,
        page_size=page_size
    )

@router.put("/{{{id_param}}}", response_model={model_name})
async def update_{operation_name}(
    id: str = Path(..., description="The ID of the {name} to update"),
    data: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_user)
):
    """Update a {name}"""
    # Check if user has permission
    requires_role(current_user, ["admin", "editor"])
    
    # Add audit fields
    data["updated_by"] = current_user.id
    data["updated_at"] = datetime.now()
    
    result = await service.update(id, data)
    if not result:
        raise HTTPException(status_code=404, detail="{model_name} not found")
    return result

@router.delete("/{{{id_param}}}")
async def delete_{operation_name}(
    id: str = Path(..., description="The ID of the {name} to delete"),
    current_user: User = Depends(get_current_user)
):
    """Delete a {name}"""
    # Check if user has permission
    requires_role(current_user, ["admin"])
    
    result = await service.delete(id)
    if not result:
        raise HTTPException(status_code=404, detail="{model_name} not found")
    return {{"success": True}}

# ----- Advanced Operations -----

@router.post("/{{{id_param}}}/analyze")
async def analyze_{operation_name}(
    id: str = Path(..., description="The ID of the {name} to analyze"),
    query: Dict[str, str] = Body(...),
    current_user: User = Depends(get_current_user)
):
    """Analyze a {name} with LLM"""
    result = await service.analyze(id, query.get("question", ""))
    return {{"result": result}}

@router.post("/bulk")
async def bulk_create_{operation_name}(
    items: List[Dict[str, Any]] = Body(...),
    current_user: User = Depends(get_current_user)
):
    """Create multiple {name}s in a single transaction"""
    # Check if user has permission
    requires_role(current_user, ["admin"])
    
    # Add audit fields to each item
    for item in items:
        item["created_by"] = current_user.id
    
    results = await service.bulk_create(items)
    return {{"created": len(results)}}

@router.get("/export")
async def export_{operation_name}(
    format: str = Query("json", description="Export format (json or csv)"),
    current_user: User = Depends(get_current_user)
):
    """Export all {name}s in specified format"""
    # Check if user has permission
    requires_role(current_user, ["admin", "editor"])
    
    data = await service.export(format)
    
    # Set appropriate content type and filename
    content_type = "application/json" if format.lower() == "json" else "text/csv"
    filename = f"{route_prefix}.{{format.lower()}}"
    
    return StreamingResponse(
        io.StringIO(data),
        media_type=content_type,
        headers={{"Content-Disposition": f"attachment; filename={{filename}}"}}
    )
"""

# Template for an advanced UI component with filtering, sorting, pagination
ADVANCED_UI_LIST_TEMPLATE = """import React, {{ useState, useEffect, useCallback }} from 'react';
import Link from 'next/link';
import {{ useRouter }} from 'next/router';

type {model_name} = {{
  id: string;
  name: string;
  slug: string;
  status: string;
  created_at: string;
  updated_at: string;
  // Add other fields as needed
}};

type SortOrder = 'asc' | 'desc';

export default function {model_name}List() {{
  const router = useRouter();
  
  // State
  const [items, setItems] = useState<{model_name}[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  
  // Filtering state
  const [statusFilter, setStatusFilter] = useState<string>('');
  
  // Sorting state
  const [sortField, setSortField] = useState<string>('created_at');
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc');
  
  // Pagination state
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [totalItems, setTotalItems] = useState(0);
  
  // Function to fetch data with current filters, sorting, and pagination
  const fetchData = useCallback(async () => {{
    setLoading(true);
    setError('');
    
    try {{
      // Build query params
      const params = new URLSearchParams();
      
      // Add filters
      if (statusFilter) params.append('status', statusFilter);
      
      // Add sorting
      params.append('sort_by', sortField);
      params.append('sort_order', sortOrder);
      
      // Add pagination
      params.append('page', page.toString());
      params.append('page_size', pageSize.toString());
      
      // Make API request
      const response = await fetch(`/api/{model_name_lower}?${{params.toString()}}`);
      
      if (response.ok) {{
        const data = await response.json();
        setItems(data);
        
        // Get total count from headers if available
        const totalCount = response.headers.get('X-Total-Count');
        if (totalCount) {{
          setTotalItems(parseInt(totalCount, 10));
        }}
      }} else {{
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to fetch data');
      }}
    }} catch (err) {{
      setError('Error fetching data');
      console.error(err);
    }} finally {{
      setLoading(false);
    }}
  }}, [statusFilter, sortField, sortOrder, page, pageSize]);
  
  // Fetch data when dependencies change
  useEffect(() => {{
    fetchData();
  }}, [fetchData]);
  
  // Toggle sort order or change sort field
  const handleSort = (field: string) => {{
    if (field === sortField) {{
      // Toggle order if same field
      setSortOrder(prevOrder => prevOrder === 'asc' ? 'desc' : 'asc');
    }} else {{
      // Set new field and default to ascending
      setSortField(field);
      setSortOrder('asc');
    }}
  }};
  
  const handleDelete = async (id: string) => {{
    if (!confirm('Are you sure you want to delete this item?')) {{
      return;
    }}
    
    try {{
      const response = await fetch(`/api/{model_name_lower}/${{id}}`, {{
        method: 'DELETE',
      }});
      
      if (response.ok) {{
        // Refresh data
        fetchData();
      }} else {{
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to delete item');
      }}
    }} catch (err) {{
      console.error('Error deleting item:', err);
      setError('Error deleting item');
    }}
  }};
  
  // Calculate total pages
  const totalPages = Math.ceil(totalItems / pageSize);
  
  // Render loading state
  if (loading && items.length === 0) return <div className="loading">Loading...</div>;
  
  // Render error
  if (error) return <div className="error">Error: {{error}}</div>;
  
  return (
    <div className="{model_name_lower}-list">
      <h2>{model_name} List</h2>
      
      {/* Filters */}
      <div className="filters">
        <div className="filter-group">
          <label htmlFor="status-filter">Status:</label>
          <select 
            id="status-filter" 
            value={{statusFilter}} 
            onChange={{(e) => setStatusFilter(e.target.value)}}
          >
            <option value="">All</option>
            <option value="active">Active</option>
            <option value="inactive">Inactive</option>
            <option value="draft">Draft</option>
          </select>
        </div>
        
        <button onClick={{() => {{
          // Clear filters
          setStatusFilter('');
          setPage(1);
        }}}}>
          Clear Filters
        </button>
      </div>
      
      {/* Results */}
      {{items.length === 0 ? (
        <p className="no-results">No items found</p>
      ) : (
        <>
          <table className="data-table">
            <thead>
              <tr>
                <th onClick={{() => handleSort('name')}}>
                  Name
                  {{sortField === 'name' && (
                    <span>{{sortOrder === 'asc' ? ' ↑' : ' ↓'}}</span>
                  )}}
                </th>
                <th onClick={{() => handleSort('status')}}>
                  Status
                  {{sortField === 'status' && (
                    <span>{{sortOrder === 'asc' ? ' ↑' : ' ↓'}}</span>
                  )}}
                </th>
                <th onClick={{() => handleSort('created_at')}}>
                  Created
                  {{sortField === 'created_at' && (
                    <span>{{sortOrder === 'asc' ? ' ↑' : ' ↓'}}</span>
                  )}}
                </th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {{items.map(item => (
                <tr key={{item.id}}>
                  <td>{{item.name}}</td>
                  <td>
                    <span className={{`status status-${{item.status}}`}}>
                      {{item.status}}
                    </span>
                  </td>
                  <td>{{new Date(item.created_at).toLocaleString()}}</td>
                  <td>
                    <div className="actions">
                      <Link href={{{{
                        pathname: '/dashboard/{model_name_lower}',
                        query: {{ mode: 'edit', id: item.id }},
                      }}}}>
                        Edit
                      </Link>
                      
                      <Link href={{`/dashboard/{model_name_lower}/${{item.id}}`}}>
                        View
                      </Link>
                      
                      <Link href={{`/dashboard/{model_name_lower}/analyze/${{item.id}}`}}>
                        Analyze
                      </Link>
                      
                      <button 
                        onClick={{() => handleDelete(item.id)}}
                        className="delete-button"
                      >
                        Delete
                      </button>
                    </div>
                  </td>
                </tr>
              ))}}
            </tbody>
          </table>
          
          {/* Pagination */}
          <div className="pagination">
            <div className="pagination-info">
              Showing {{(page - 1) * pageSize + 1}} to {{Math.min(page * pageSize, totalItems)}} of {{totalItems}} items
            </div>
            
            <div className="pagination-controls">
              <button 
                onClick={{() => setPage(1)}}
                disabled={{page === 1}}
              >
                First
              </button>
              
              <button 
                onClick={{() => setPage(p => Math.max(1, p - 1))}}
                disabled={{page === 1}}
              >
                Previous
              </button>
              
              <span className="page-info">Page {{page}} of {{totalPages}}</span>
              
              <button 
                onClick={{() => setPage(p => Math.min(totalPages, p + 1))}}
                disabled={{page === totalPages}}
              >
                Next
              </button>
              
              <button 
                onClick={{() => setPage(totalPages)}}
                disabled={{page === totalPages}}
              >
                Last
              </button>
            </div>
            
            <div className="page-size-control">
              <label htmlFor="page-size">Items per page:</label>
              <select 
                id="page-size"
                value={{pageSize}} 
                onChange={{(e) => {{
                  setPageSize(Number(e.target.value));
                  setPage(1); // Reset to first page when changing page size
                }}}}
              >
                <option value="5">5</option>
                <option value="10">10</option>
                <option value="25">25</option>
                <option value="50">50</option>
                <option value="100">100</option>
              </select>
            </div>
          </div>
        </>
      )}}
    </div>
  );
}}
"""

# Template for an advanced form component with validation
ADVANCED_UI_FORM_TEMPLATE = """import React, {{ useState, useEffect, useCallback }} from 'react';
import {{ useForm, SubmitHandler }} from 'react-hook-form';
import {{ yupResolver }} from '@hookform/resolvers/yup';
import * as yup from 'yup';

// Define form schema
const schema = yup.object({{
  name: yup.string().required('Name is required'),
  slug: yup.string(),
  description: yup.string(),
  status: yup.string().oneOf(['active', 'inactive', 'draft'], 'Invalid status'),
  // Add additional field validations here
}}).required();

// Form field types
type {model_name}FormValues = {{
  name: string;
  slug?: string;
  description?: string;
  status: string;
  // Add other fields as needed
}};

type {model_name}FormProps = {{
  id?: string;
  onSubmit: (data: {model_name}FormValues) => void;
  onCancel?: () => void;
}};

export default function {model_name}Form({{ id, onSubmit, onCancel }}: {model_name}FormProps) {{
  // Set up form with validation
  const {{ 
    register, 
    handleSubmit, 
    reset, 
    formState: {{ errors, isSubmitting, isDirty, isValid }},
    setValue,
    watch
  }} = useForm<{model_name}FormValues>({{
    resolver: yupResolver(schema),
    mode: 'onChange',
    defaultValues: {{
      name: '',
      description: '',
      status: 'active',
      // Add other default values
    }}
  }});
  
  const [serverError, setServerError] = useState('');
  const [loadingData, setLoadingData] = useState(id ? true : false);
  
  // Watch the name field to provide slug suggestion
  const nameValue = watch('name');
  
  // Function to generate slug from name
  const generateSlug = useCallback((name: string) => {{
    return name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '');
  }}, []);
  
  // Auto-suggest slug when name changes and slug is empty
  useEffect(() => {{
    if (nameValue && !watch('slug')) {{
      setValue('slug', generateSlug(nameValue), {{ shouldValidate: true }});
    }}
  }}, [nameValue, setValue, generateSlug, watch]);
  
  // Fetch existing data if editing
  useEffect(() => {{
    if (id) {{
      const fetchData = async () => {{
        try {{
          setLoadingData(true);
          setServerError('');
          
          const response = await fetch(`/api/{model_name_lower}/${{id}}`);
          
          if (response.ok) {{
            const data = await response.json();
            
            // Set form values from response
            reset(data);
          }} else {{
            const errorData = await response.json();
            setServerError(errorData.detail || 'Failed to load data');
          }}
        }} catch (err) {{
          setServerError('Error loading data');
          console.error(err);
        }} finally {{
          setLoadingData(false);
        }}
      }};
      
      fetchData();
    }}
  }}, [id, reset]);
  
  // Form submission handler
  const processSubmit: SubmitHandler<{model_name}FormValues> = async (data) => {{
    try {{
      // Clear any previous errors
      setServerError('');
      
      // Call the provided onSubmit callback
      await onSubmit(data);
    }} catch (err) {{
      console.error('Error submitting form:', err);
      setServerError('Error submitting form');
    }}
  }};
  
  if (loadingData) {{
    return <div className="loading">Loading...</div>;
  }}
  
  return (
    <form 
      onSubmit={{handleSubmit(processSubmit)}} 
      className="{model_name_lower}-form"
    >
      {{serverError && (
        <div className="error-message">{{serverError}}</div>
      )}}
      
      <div className="form-row">
        <div className="form-group">
          <label htmlFor="name">
            Name <span className="required">*</span>
          </label>
          <input
            id="name"
            type="text"
            {{...register('name')}}
            className={{errors.name ? 'has-error' : ''}}
          />
          {{errors.name && (
            <div className="field-error">{{errors.name.message}}</div>
          )}}
        </div>
        
        <div className="form-group">
          <label htmlFor="slug">Slug</label>
          <div className="input-with-button">
            <input
              id="slug"
              type="text"
              {{...register('slug')}}
              className={{errors.slug ? 'has-error' : ''}}
              placeholder="auto-generated"
            />
            <button
              type="button"
              onClick={{() => setValue('slug', generateSlug(nameValue), {{ shouldValidate: true }})}}
              className="secondary-button"
            >
              Generate
            </button>
          </div>
          {{errors.slug && (
            <div className="field-error">{{errors.slug.message}}</div>
          )}}
          <div className="field-help">Leave empty to auto-generate from name</div>
        </div>
      </div>
      
      <div className="form-group">
        <label htmlFor="description">Description</label>
        <textarea
          id="description"
          {{...register('description')}}
          rows={{4}}
          className={{errors.description ? 'has-error' : ''}}
        />
        {{errors.description && (
          <div className="field-error">{{errors.description.message}}</div>
        )}}
      </div>
      
      <div className="form-group">
        <label htmlFor="status">Status</label>
        <select
          id="status"
          {{...register('status')}}
          className={{errors.status ? 'has-error' : ''}}
        >
          <option value="active">Active</option>
          <option value="inactive">Inactive</option>
          <option value="draft">Draft</option>
        </select>
        {{errors.status && (
          <div className="field-error">{{errors.status.message}}</div>
        )}}
      </div>
      
      {/* Add more form fields as needed */}
      
      <div className="form-actions">
        {{onCancel && (
          <button 
            type="button" 
            onClick={{onCancel}}
            className="cancel-button"
          >
            Cancel
          </button>
        )}}
        
        <button 
          type="submit"
          disabled={{isSubmitting || (!isDirty && !id)}}
          className="submit-button"
        >
          {{isSubmitting ? 'Saving...' : id ? 'Update' : 'Create'}}
        </button>
      </div>
    </form>
  );
}}
"""

# Template for a comprehensive test suite
ADVANCED_TEST_TEMPLATE = """import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient
from fastapi import FastAPI
import json
from datetime import datetime, timedelta

from fortitude.backend.models.{model_file} import {model_name}
from fortitude.backend.services.{service_file} import {service_name}
from fortitude.backend.controllers.{controller_file} import router

# Create test app
app = FastAPI()
app.include_router(router)

# Mock the auth dependency
@pytest.fixture
def mock_auth():
    with patch('fortitude.backend.auth.get_current_user') as mock:
        mock.return_value = MagicMock(
            id="test-user-id",
            username="testuser",
            roles=["admin"]
        )
        yield mock

# Test data
@pytest.fixture
def sample_data():
    return {{
        "name": "Test {model_name}",
        "description": "Test description",
        "status": "active"
    }}

# Model tests
class Test{model_name}Model:
    def test_creation(self):
        """Test {model_name} creation with minimum fields"""
        instance = {model_name}(name="Test {model_name}")
        
        assert instance.name == "Test {model_name}"
        assert instance.status == "active"  # Default value
        assert instance.id is not None
    
    def test_slug_generation(self):
        """Test automatic slug generation"""
        instance = {model_name}(name="Test {model_name}")
        
        assert instance.slug == "test-{model_name_lower}"
    
    def test_is_active_property(self):
        """Test is_active computed property"""
        active = {model_name}(name="Active", status="active")
        inactive = {model_name}(name="Inactive", status="inactive")
        
        assert active.is_active is True
        assert inactive.is_active is False

# Service tests
class Test{service_name}:
    @pytest.fixture
    def service(self):
        return {service_name}()
    
    @pytest.mark.asyncio
    async def test_create(self, service, sample_data):
        """Test creating a {model_name}"""
        instance = await service.create(sample_data)
        
        assert instance.name == sample_data["name"]
        assert instance.description == sample_data["description"]
        assert instance.status == sample_data["status"]
        assert instance.id is not None
        
        # Verify it was stored in the service
        assert instance.id in service._data
    
    @pytest.mark.asyncio
    async def test_get(self, service, sample_data):
        """Test getting a {model_name} by ID"""
        # First create an instance
        created = await service.create(sample_data)
        
        # Then retrieve it
        retrieved = await service.get(created.id)
        
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.name == created.name
    
    @pytest.mark.asyncio
    async def test_list_with_filters(self, service):
        """Test listing {model_name}s with filters"""
        # Create test instances
        await service.create({{"name": "Active Item", "status": "active"}})
        await service.create({{"name": "Inactive Item", "status": "inactive"}})
        await service.create({{"name": "Another Active", "status": "active"}})
        
        # Filter by status
        active_items = await service.list(filters={{"status": "active"}})
        
        assert len(active_items) == 2
        assert all(item.status == "active" for item in active_items)
    
    @pytest.mark.asyncio
    async def test_update(self, service, sample_data):
        """Test updating a {model_name}"""
        # First create an instance
        created = await service.create(sample_data)
        
        # Then update it
        updated = await service.update(created.id, {{"name": "Updated Name"}})
        
        assert updated is not None
        assert updated.id == created.id
        assert updated.name == "Updated Name"
        assert updated.status == created.status  # Unchanged
    
    @pytest.mark.asyncio
    async def test_delete(self, service, sample_data):
        """Test deleting a {model_name}"""
        # First create an instance
        created = await service.create(sample_data)
        
        # Verify it exists
        assert await service.get(created.id) is not None
        
        # Delete it
        result = await service.delete(created.id)
        
        assert result is True
        assert await service.get(created.id) is None

# API endpoint tests
class TestAPI:
    @pytest.mark.asyncio
    async def test_create_endpoint(self, mock_auth, sample_data):
        """Test creating a {model_name} via API"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/{route_prefix}", json=sample_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == sample_data["name"]
            assert "id" in data
    
    @pytest.mark.asyncio
    async def test_get_endpoint(self, mock_auth, sample_data):
        """Test getting a {model_name} via API"""
        # Mock service.get to return a {model_name}
        with patch('fortitude.backend.services.{service_file}.{service_name}.get') as mock_get:
            mock_instance = {model_name}(**sample_data, id="test-id")
            mock_get.return_value = mock_instance
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/{route_prefix}/test-id")
                
                assert response.status_code == 200
                data = response.json()
                assert data["id"] == "test-id"
                assert data["name"] == sample_data["name"]
    
    @pytest.mark.asyncio
    async def test_list_endpoint(self, mock_auth):
        """Test listing {model_name}s via API"""
        # Mock service.list to return a list of {model_name}s
        with patch('fortitude.backend.services.{service_file}.{service_name}.list') as mock_list:
            mock_instances = [
                {model_name}(name="Item 1", id="id1"),
                {model_name}(name="Item 2", id="id2")
            ]
            mock_list.return_value = mock_instances
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/{route_prefix}")
                
                assert response.status_code == 200
                data = response.json()
                assert len(data) == 2
                assert data[0]["id"] == "id1"
                assert data[1]["id"] == "id2"
    
    @pytest.mark.asyncio
    async def test_update_endpoint(self, mock_auth):
        """Test updating a {model_name} via API"""
        update_data = {{"name": "Updated via API"}}
        
        # Mock service.update to return an updated {model_name}
        with patch('fortitude.backend.services.{service_file}.{service_name}.update') as mock_update:
            mock_instance = {model_name}(name="Updated via API", id="test-id")
            mock_update.return_value = mock_instance
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.put("/{route_prefix}/test-id", json=update_data)
                
                assert response.status_code == 200
                data = response.json()
                assert data["id"] == "test-id"
                assert data["name"] == "Updated via API"
    
    @pytest.mark.asyncio
    async def test_delete_endpoint(self, mock_auth):
        """Test deleting a {model_name} via API"""
        # Mock service.delete to return True
        with patch('fortitude.backend.services.{service_file}.{service_name}.delete') as mock_delete:
            mock_delete.return_value = True
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.delete("/{route_prefix}/test-id")
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
    
    @pytest.mark.asyncio
    async def test_not_found(self, mock_auth):
        """Test 404 response when {model_name} not found"""
        # Mock service.get to return None
        with patch('fortitude.backend.services.{service_file}.{service_name}.get') as mock_get:
            mock_get.return_value = None
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/{route_prefix}/nonexistent")
                
                assert response.status_code == 404
                data = response.json()
                assert "detail" in data
"""

# Template for deployment configurations
DEPLOYMENT_TEMPLATE = """# Docker Compose configuration for {model_name} service
version: '3'

services:
  # API Server
  {service_name_lower}:
    build: 
      context: .
      dockerfile: Dockerfile
    image: {service_name_lower}:latest
    container_name: {service_name_lower}
    restart: unless-stopped
    volumes:
      - .:/app
    environment:
      - PORT=9997
      - DATABASE_URL=postgresql://${DB_USER}:${DB_PASSWORD}@{service_name_lower}-db:5432/${DB_NAME}
      - SECRET_KEY=${SECRET_KEY}
      - ENVIRONMENT=production
    ports:
      - "9997:9997"
    depends_on:
      - {service_name_lower}-db
    networks:
      - {service_name_lower}-network
    command: ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9997"]

  # UI Server
  {service_name_lower}-ui:
    build: 
      context: ./ui
      dockerfile: Dockerfile
    image: {service_name_lower}-ui:latest
    container_name: {service_name_lower}-ui
    restart: unless-stopped
    volumes:
      - ./ui:/app
    environment:
      - API_URL=http://{service_name_lower}:9997
      - PORT=9996
    ports:
      - "9996:9996"
    depends_on:
      - {service_name_lower}
    networks:
      - {service_name_lower}-network
    command: ["npm", "run", "start"]

  # Database
  {service_name_lower}-db:
    image: postgres:14-alpine
    container_name: {service_name_lower}-db
    restart: unless-stopped
    environment:
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=${DB_NAME}
    volumes:
      - {service_name_lower}-db-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - {service_name_lower}-network

  # Redis for caching
  {service_name_lower}-redis:
    image: redis:alpine
    container_name: {service_name_lower}-redis
    restart: unless-stopped
    volumes:
      - {service_name_lower}-redis-data:/data
    networks:
      - {service_name_lower}-network

networks:
  {service_name_lower}-network:
    driver: bridge

volumes:
  {service_name_lower}-db-data:
  {service_name_lower}-redis-data:
"""

# GraphQL Schema Templates
GRAPHQL_SCHEMA_TEMPLATE = """# GraphQL Schema for {model_name}
type {model_name} {{
  id: ID!
  name: String!
{fields}
  createdAt: String!
  updatedAt: String!
}}

input {model_name}Input {{
  name: String!
{input_fields}
}}

type Query {{
  get{model_name}(id: ID!): {model_name}
  list{model_name}s: [{model_name}!]!
  search{model_name}s(query: String): [{model_name}!]!
}}

type Mutation {{
  create{model_name}(input: {model_name}Input!): {model_name}!
  update{model_name}(id: ID!, input: {model_name}Input!): {model_name}!
  delete{model_name}(id: ID!): Boolean!
}}
"""

GRAPHQL_RESOLVER_TEMPLATE = """# GraphQL Resolvers for {model_name}
from typing import List, Dict, Any, Optional
from backend.models.{model_file} import {model_name}
from backend.services.{service_file} import {service_name}

service = {service_name}()

async def get_{model_name_lower}(parent, info, id):
    """Get a {model_name} by ID"""
    return await service.get(id)

async def list_{model_name_lower}s(parent, info):
    """List all {model_name}s"""
    return await service.list()

async def search_{model_name_lower}s(parent, info, query):
    """Search for {model_name}s"""
    return await service.search(query)

async def create_{model_name_lower}(parent, info, input):
    """Create a new {model_name}"""
    return await service.create(input)

async def update_{model_name_lower}(parent, info, id, input):
    """Update a {model_name}"""
    return await service.update(id, input)

async def delete_{model_name_lower}(parent, info, id):
    """Delete a {model_name}"""
    return await service.delete(id)

resolvers = {{
    "Query": {{
        f"get{model_name}": get_{model_name_lower},
        f"list{model_name}s": list_{model_name_lower}s,
        f"search{model_name}s": search_{model_name_lower}s,
    }},
    "Mutation": {{
        f"create{model_name}": create_{model_name_lower},
        f"update{model_name}": update_{model_name_lower},
        f"delete{model_name}": delete_{model_name_lower},
    }}
}}
"""

# Domain-Driven Design Templates
DDD_ENTITY_TEMPLATE = """from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

@dataclass
class {entity_name}:
    """"{entity_name} entity in the {domain_name} domain"""
{fields}
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def update(self, **kwargs):
        """Update entity attributes"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.now()
"""

DDD_REPOSITORY_TEMPLATE = """from typing import List, Optional, Dict, Any
from .entities import {entity_name}

class {entity_name}Repository:
    """Repository for {entity_name} entities"""
    
    def __init__(self):
        self._entities: Dict[str, {entity_name}] = {{}}
    
    async def save(self, entity: {entity_name}) -> {entity_name}:
        """Save an entity"""
        self._entities[entity.id] = entity
        return entity
    
    async def find_by_id(self, id: str) -> Optional[{entity_name}]:
        """Find an entity by ID"""
        return self._entities.get(id)
    
    async def find_all(self) -> List[{entity_name}]:
        """Find all entities"""
        return list(self._entities.values())
    
    async def delete(self, id: str) -> bool:
        """Delete an entity"""
        if id in self._entities:
            del self._entities[id]
            return True
        return False
"""

DDD_SERVICE_TEMPLATE = """from typing import List, Optional, Dict, Any
from .repositories import {entity_name}Repository
from .entities import {entity_name}
from .value_objects import {value_objects_import}
from .events import {event_bus_import}

class {entity_name}Service:
    """Service for {entity_name} operations"""
    
    def __init__(
        self, 
        repository: {entity_name}Repository,
        event_bus: {event_bus_class}
    ):
        self.repository = repository
        self.event_bus = event_bus
    
    async def create(self, data: Dict[str, Any]) -> {entity_name}:
        """Create a new {entity_name}"""
        entity = {entity_name}(**data)
        saved = await self.repository.save(entity)
        
        # Publish domain event
        await self.event_bus.publish(f"{entity_name}Created", {{"id": saved.id}})
        
        return saved
    
    async def get(self, id: str) -> Optional[{entity_name}]:
        """Get an {entity_name} by ID"""
        return await self.repository.find_by_id(id)
    
    async def list(self) -> List[{entity_name}]:
        """List all {entity_name}s"""
        return await self.repository.find_all()
    
    async def update(self, id: str, data: Dict[str, Any]) -> Optional[{entity_name}]:
        """Update an {entity_name}"""
        entity = await self.repository.find_by_id(id)
        if not entity:
            return None
        
        entity.update(**data)
        saved = await self.repository.save(entity)
        
        # Publish domain event
        await self.event_bus.publish(f"{entity_name}Updated", {{"id": saved.id}})
        
        return saved
    
    async def delete(self, id: str) -> bool:
        """Delete an {entity_name}"""
        success = await self.repository.delete(id)
        
        if success:
            # Publish domain event
            await self.event_bus.publish(f"{entity_name}Deleted", {{"id": id}})
        
        return success
"""

DDD_VALUE_OBJECT_TEMPLATE = """from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class {name}:
    """"{name} value object in the {domain_name} domain"""
{fields}
    
    def __post_init__(self):
        """Validate the value object"""
        # Add validation logic here
        pass
"""

DDD_EVENT_BUS_TEMPLATE = """from typing import Dict, Any, Callable, List, Awaitable
import asyncio

class EventBus:
    """Event bus for domain events"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable[[Dict[str, Any]], Awaitable[None]]]] = {{}}
    
    def subscribe(self, event_type: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]):
        """Subscribe to an event"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    async def publish(self, event_type: str, event_data: Dict[str, Any]):
        """Publish an event"""
        if event_type in self.subscribers:
            await asyncio.gather(
                *[handler(event_data) for handler in self.subscribers[event_type]]
            )
"""

# Microservice Templates
MICROSERVICE_MAIN_TEMPLATE = """#!/usr/bin/env python3

import os
import asyncio
import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import routers
{router_imports}

# Import from shared modules
from {service_name_lower}.config import settings
from {service_name_lower}.database import init_db, close_db
from {service_name_lower}.middleware import add_middleware

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the application"""
    # Startup
    await init_db()
    
    yield
    
    # Shutdown
    await close_db()

# Create application
app = FastAPI(
    title="{service_name} Microservice",
    description="{service_description}",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
add_middleware(app)

# Add routers
{router_registrations}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
"""

MICROSERVICE_CONFIG_TEMPLATE = """#!/usr/bin/env python3

import os
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    """Application settings"""
    
    # General
    APP_NAME: str = "{service_name}"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    ENV: str = os.getenv("ENV", "development")
    
    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "{port}"))
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:9996",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:9996",
    ]
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./app.db")
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "development_secret_key")
    TOKEN_EXPIRE_MINUTES: int = int(os.getenv("TOKEN_EXPIRE_MINUTES", "60"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
"""

MICROSERVICE_DATABASE_TEMPLATE = """#!/usr/bin/env python3

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.asyncio import AsyncEngine
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from {service_name_lower}.config import settings

# Create async engine
engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True
)

# Create async session factory
async_session = sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# Create declarative base for models
Base = declarative_base()

async def init_db():
    """Initialize database"""
    if settings.ENV == "development":
        async with engine.begin() as conn:
            # Create tables for development
            await conn.run_sync(Base.metadata.create_all)

async def close_db():
    """Close database connections"""
    await engine.dispose()

@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session"""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
"""

# Dockerfile Template
DOCKERFILE_TEMPLATE = """# Dockerfile for {model_name} service
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=9997

# Expose port
EXPOSE ${PORT}

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT}"]
"""

UI_DOCKERFILE_TEMPLATE = """# Dockerfile for {model_name} UI
FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json ./
RUN npm ci

# Copy application code
COPY . .

# Build the application
RUN npm run build

# Expose port
EXPOSE 9996

# Start the application
CMD ["npm", "run", "start"]
"""

def generate_advanced_scaffold_config(name: str, output_path: str) -> str:
    """Generate an advanced scaffold configuration file
    
    Args:
        name: The name of the scaffold
        output_path: The output path for the config
        
    Returns:
        The path to the generated config file
    """
    model_name = name.capitalize()
    model_name_lower = name.lower()
    
    content = f"""#!/usr/bin/env python3
\"\"\"Advanced scaffold configuration for {name}

This file defines the structure and components for the {name} advanced scaffold.
The configuration provides detailed specifications for models, endpoints, UI components,
tests, deployment, MCP integration, authentication, caching, and database configuration.
\"\"\"

# Define the models to be created - these include full field definitions with validation rules
models = [
    {{
        "name": "{model_name}",
        "docstring": "Advanced {model_name} model with relationships, validations and computed properties",
        "fields": [
            # Basic information fields
            {{"name": "name", "type": "str", "required": True, "description": "Name of the {model_name}"}},
            {{"name": "description", "type": "str", "required": False, "description": "Detailed description of the {model_name}"}},
            {{"name": "code", "type": "str", "required": True, "description": "Unique identifier code for the {model_name}", "validation": "regex(r'^[A-Z]{{3}}-\\d{{4}}$')"}},
            
            # Numerical fields with validation
            {{"name": "price", "type": "float", "required": True, "description": "Price in USD", "validation": "ge(0.0)"}},
            {{"name": "discount_percent", "type": "float", "required": False, "default": 0.0, "description": "Discount percentage", "validation": "ge(0.0) and le(100.0)"}},
            {{"name": "quantity", "type": "int", "required": True, "default": 0, "description": "Available quantity", "validation": "ge(0)"}},
            {{"name": "min_order", "type": "int", "required": False, "default": 1, "description": "Minimum order quantity", "validation": "ge(1)"}},
            {{"name": "max_order", "type": "int", "required": False, "description": "Maximum order quantity", "validation": "gt(field('min_order'))"}},
            
            # Categorization fields
            {{"name": "category", "type": "str", "required": False, "description": "Primary category", "validation": "in_(['electronics', 'clothing', 'food', 'books', 'other'])"}},
            {{"name": "subcategory", "type": "str", "required": False, "description": "Sub-category for more specific classification"}},
            {{"name": "tags", "type": "List[str]", "required": False, "default": "[]", "description": "List of tags for searchability"}},
            
            # Complex fields with nested structures
            {{"name": "dimensions", "type": "Dict[str, float]", "required": False, "default": "{{'length': 0.0, 'width': 0.0, 'height': 0.0}}", "description": "Physical dimensions in inches"}},
            {{"name": "metadata", "type": "Dict[str, Any]", "required": False, "default": "{{}}", "description": "Additional metadata for extensibility"}},
            {{"name": "images", "type": "List[Dict[str, str]]", "required": False, "default": "[]", "description": "List of image objects with URL and caption"}},
            
            # Date and time fields
            {{"name": "release_date", "type": "datetime", "required": False, "description": "Product release date"}},
            {{"name": "discontinued", "type": "bool", "required": False, "default": "False", "description": "Whether the product is discontinued"}},
        ],
        "relations": [
            # Ownership relations
            {{"name": "owner_id", "type": "str", "required": True, "docstring": "ID of the user who owns this {model_name}"}},
            {{"name": "creator_id", "type": "str", "required": True, "docstring": "ID of the user who created this {model_name}"}},
            
            # One-to-many relations
            {{"name": "supplier_id", "type": "str", "required": False, "docstring": "ID of the supplier for this {model_name}"}},
            {{"name": "manufacturer_id", "type": "str", "required": False, "docstring": "ID of the manufacturer for this {model_name}"}},
            
            # Many-to-many relations
            {{"name": "related_ids", "type": "List[str]", "required": False, "default": "[]", "docstring": "IDs of related {model_name}s"}},
            {{"name": "alternative_ids", "type": "List[str]", "required": False, "default": "[]", "docstring": "IDs of alternative {model_name}s"}},
            
            # Complex relations with metadata
            {{"name": "category_metadata", "type": "Dict[str, Any]", "required": False, "default": "{{}}", "docstring": "Additional metadata for category relations"}},
        ],
        "computed_properties": [
            {{"name": "discounted_price", "type": "float", "description": "Price after applying discount", "formula": "self.price * (1 - self.discount_percent / 100)"}},
            {{"name": "is_in_stock", "type": "bool", "description": "Whether the product is in stock", "formula": "self.quantity > 0"}},
            {{"name": "full_name", "type": "str", "description": "Full product name with code", "formula": "f'{{self.name}} ({{self.code}})'"}},
        ],
        "indexes": [
            "name", "code", "category", "supplier_id", "created_at", "price"
        ],
        "constraints": [
            {{"type": "unique", "fields": ["code"]}},
            {{"type": "unique", "fields": ["name", "owner_id"]}},
            {{"type": "check", "condition": "quantity >= 0"}}
        ]
    }},
    
    # Additional related models
    {{
        "name": "Category",
        "docstring": "Category for {model_name}s with hierarchical structure",
        "fields": [
            {{"name": "name", "type": "str", "required": True, "description": "Name of the category"}},
            {{"name": "slug", "type": "str", "required": True, "description": "URL-friendly slug", "validation": "regex(r'^[a-z0-9-]+$')"}},
            {{"name": "description", "type": "str", "required": False, "description": "Description of the category"}},
            {{"name": "icon", "type": "str", "required": False, "description": "Icon name for the category"}},
            {{"name": "display_order", "type": "int", "required": False, "default": 0, "description": "Display order for sorting"}},
        ],
        "relations": [
            {{"name": "parent_id", "type": "str", "required": False, "docstring": "Parent category ID for hierarchical categories"}},
            {{"name": "child_ids", "type": "List[str]", "required": False, "default": "[]", "docstring": "IDs of child categories"}},
        ]
    }},
    
    {{
        "name": "Review",
        "docstring": "Customer review for a {model_name}",
        "fields": [
            {{"name": "title", "type": "str", "required": True, "description": "Review title"}},
            {{"name": "content", "type": "str", "required": True, "description": "Review content"}},
            {{"name": "rating", "type": "int", "required": True, "description": "Rating from 1-5", "validation": "ge(1) and le(5)"}},
            {{"name": "verified_purchase", "type": "bool", "required": False, "default": "False", "description": "Whether the review is from a verified purchase"}},
            {{"name": "helpful_votes", "type": "int", "required": False, "default": 0, "description": "Number of helpful votes"}},
        ],
        "relations": [
            {{"name": "product_id", "type": "str", "required": True, "docstring": "ID of the reviewed {model_name}"}},
            {{"name": "user_id", "type": "str", "required": True, "docstring": "ID of the user who wrote the review"}},
        ]
    }}
]

# Define the endpoints to be created with full CRUD operations and specialized endpoints
endpoints = [
    {{
        "name": "{model_name_lower}",
        "model": "{model_name}",
        "operations": [
            "create",                # Basic CRUD operations
            "read", 
            "update", 
            "delete", 
            "list",
            
            "analyze",                # Advanced analytics with LLM
            "export",                 # Data export in multiple formats
            "bulk_create",           # Bulk operations for efficiency
            "bulk_update",
            
            "search",                # Advanced search capabilities
            "filter",                # Complex filtering
            "count",                 # Aggregation endpoints
            
            "recommend",             # AI-powered recommendations
            "similar",               # Find similar items
            
            "upvote",                # User interactions
            "downvote",
            "favorite",
            
            "publish",               # Workflow operations
            "unpublish",
            "archive"
        ],
        "middlewares": [
            "authentication",        # Security middlewares
            "authorization",
            "rate_limiting",
            "caching",              # Performance middlewares
            "logging",              # Observability
            "validation"            # Data validation
        ],
        "query_params": [
            {{"name": "status", "type": "str", "description": "Filter by status", "values": ["active", "inactive", "draft", "archived"]}},
            {{"name": "category", "type": "str", "description": "Filter by category"}},
            {{"name": "min_price", "type": "float", "description": "Minimum price filter"}},
            {{"name": "max_price", "type": "float", "description": "Maximum price filter"}},
            {{"name": "sort_by", "type": "str", "description": "Field to sort by", "values": ["name", "price", "created_at", "updated_at", "rating"]}},
            {{"name": "sort_order", "type": "str", "description": "Sort direction", "values": ["asc", "desc"]}},
            {{"name": "page", "type": "int", "description": "Page number for pagination", "default": 1}},
            {{"name": "page_size", "type": "int", "description": "Items per page", "default": 20, "max": 100}},
            {{"name": "search", "type": "str", "description": "Search term for text search"}},
            {{"name": "fields", "type": "str", "description": "Comma-separated list of fields to include"}},
            {{"name": "expand", "type": "str", "description": "Comma-separated list of relations to expand"}},
            {{"name": "format", "type": "str", "description": "Response format", "values": ["json", "csv", "xlsx"]}},
        ]
    }},
    
    # Related model endpoints
    {{
        "name": "category",
        "model": "Category",
        "operations": ["create", "read", "update", "delete", "list", "tree", "count"],
    }},
    
    {{
        "name": "review",
        "model": "Review",
        "operations": ["create", "read", "update", "delete", "list", "approve", "reject", "report"],
    }},
    
    # Aggregation and statistics endpoints
    {{
        "name": "statistics",
        "model": "{model_name}",
        "operations": ["product_stats", "category_distribution", "price_histogram", "sales_trends"],
    }}
]

# Define the UI components to be created with detailed component specifications
ui_components = [
    # Pages
    {{
        "type": "page",
        "path": "app/dashboard/{model_name_lower}.tsx",
        "model": "{model_name}",
        "components": ["list", "form", "detail", "analyze", "statistics"],
        "layout": "dashboard",
        "title": "{model_name} Dashboard",
        "description": "Manage and analyze all {model_name}s",
        "permissions": ["view_{model_name_lower}s", "manage_{model_name_lower}s"]
    }},
    {{
        "type": "page",
        "path": "app/dashboard/{model_name_lower}/new.tsx",
        "model": "{model_name}",
        "components": ["form"],
        "layout": "dashboard",
        "title": "Create New {model_name}",
        "description": "Create a new {model_name} with all details",
        "permissions": ["create_{model_name_lower}"]
    }},
    {{
        "type": "page",
        "path": "app/dashboard/{model_name_lower}/[id].tsx",
        "model": "{model_name}",
        "components": ["detail", "related_items", "activity_log"],
        "layout": "dashboard",
        "title": "{model_name} Details",
        "description": "View and edit {model_name} details",
        "permissions": ["view_{model_name_lower}"]
    }},
    {{
        "type": "page",
        "path": "app/dashboard/{model_name_lower}/[id]/edit.tsx",
        "model": "{model_name}",
        "components": ["form"],
        "layout": "dashboard",
        "title": "Edit {model_name}",
        "description": "Edit an existing {model_name}",
        "permissions": ["edit_{model_name_lower}"]
    }},
    {{
        "type": "page",
        "path": "app/dashboard/{model_name_lower}/[id]/analyze.tsx",
        "model": "{model_name}",
        "components": ["analyze", "statistics", "insights"],
        "layout": "dashboard",
        "title": "Analyze {model_name}",
        "description": "Get AI-powered insights about this {model_name}",
        "permissions": ["view_{model_name_lower}"]
    }},
    {{
        "type": "page",
        "path": "app/dashboard/statistics/{model_name_lower}.tsx",
        "model": "{model_name}",
        "components": ["statistics", "charts", "filters"],
        "layout": "dashboard",
        "title": "{model_name} Statistics",
        "description": "Comprehensive statistics and charts for {model_name}s",
        "permissions": ["view_statistics"]
    }},
    
    # List component with advanced features
    {{
        "type": "component",
        "path": "components/{model_name_lower}/{model_name}List.tsx",
        "model": "{model_name}",
        "advanced": True,
        "features": [
            "filtering",             # Data manipulation features
            "sorting",
            "pagination",
            "search",
            "bulk_actions",          # Batch operations
            "column_customization",  # UI customization
            "export",                # Data export
            "view_switching",        # View as table, grid, or list
            "saved_views",           # Save custom views
            "row_selection",         # Select rows for actions
            "inline_editing",        # Edit without leaving the list
            "drag_and_drop",         # Reorder items
            "infinite_scroll",       # Load more items as needed
            "virtualization"         # Efficient rendering of large lists
        ],
        "filter_fields": ["name", "category", "price", "quantity", "status", "created_at"],
        "search_fields": ["name", "description", "code"],
        "bulk_actions": ["delete", "export", "change_status", "change_category"],
        "view_types": ["table", "grid", "list", "kanban"]
    }},
    
    # Form component with validation and advanced features
    {{
        "type": "component",
        "path": "components/{model_name_lower}/{model_name}Form.tsx",
        "model": "{model_name}",
        "advanced": True,
        "features": [
            "validation",           # Form validation
            "error_handling",       # Error handling
            "wizard",               # Multi-step form
            "conditional_fields",   # Show/hide fields based on conditions
            "field_dependencies",   # Update fields based on other fields
            "autosave",             # Automatically save drafts
            "form_sections",        # Organize fields into sections
            "rich_inputs",          # Rich input types
            "file_uploads",         # File upload support
            "suggestions",          # Field value suggestions
            "templates",            # Form templates
            "history",              # Edit history
            "preview"               # Live preview
        ],
        "sections": [
            {{"name": "basic", "title": "Basic Information", "fields": ["name", "description", "code"]}},
            {{"name": "pricing", "title": "Pricing", "fields": ["price", "discount_percent"]}},
            {{"name": "inventory", "title": "Inventory", "fields": ["quantity", "min_order", "max_order"]}},
            {{"name": "categorization", "title": "Categorization", "fields": ["category", "subcategory", "tags"]}},
            {{"name": "relationships", "title": "Relationships", "fields": ["owner_id", "supplier_id", "related_ids"]}},
            {{"name": "metadata", "title": "Additional Information", "fields": ["dimensions", "metadata", "images"]}}
        ],
        "validation_rules": [
            {{"field": "name", "rules": ["required", "min:3", "max:100"]}},
            {{"field": "code", "rules": ["required", "pattern:^[A-Z]{{3}}-\\d{{4}}$", "unique"]}},
            {{"field": "price", "rules": ["required", "min:0"]}},
            {{"field": "quantity", "rules": ["required", "min:0", "integer"]}}
        ]
    }},
    
    # Detail component with all information and actions
    {{
        "type": "component",
        "path": "components/{model_name_lower}/{model_name}Detail.tsx",
        "model": "{model_name}",
        "advanced": True,
        "features": [
            "tabs",                 # Tabbed interface
            "related_items",        # Show related items
            "activity_log",         # Show activity history
            "comments",             # User comments
            "share",                # Share options
            "export",               # Export options
            "print",                # Print view
            "versions",             # Version history
            "status_badges",        # Visual status indicators
            "actions_menu",         # Actions menu
            "attachments",          # File attachments
            "quick_edit"            # Edit without leaving the detail view
        ],
        "tabs": [
            {{"name": "overview", "title": "Overview", "icon": "info-circle"}},
            {{"name": "details", "title": "Details", "icon": "list-alt"}},
            {{"name": "images", "title": "Images", "icon": "images"}},
            {{"name": "pricing", "title": "Pricing", "icon": "tag"}},
            {{"name": "inventory", "title": "Inventory", "icon": "box"}},
            {{"name": "related", "title": "Related Items", "icon": "link"}},
            {{"name": "reviews", "title": "Reviews", "icon": "star"}},
            {{"name": "history", "title": "History", "icon": "history"}}
        ],
        "actions": [
            {{"name": "edit", "title": "Edit", "icon": "edit", "permission": "edit_{model_name_lower}"}},
            {{"name": "delete", "title": "Delete", "icon": "trash", "permission": "delete_{model_name_lower}"}},
            {{"name": "duplicate", "title": "Duplicate", "icon": "copy", "permission": "create_{model_name_lower}"}},
            {{"name": "export", "title": "Export", "icon": "download", "permission": "export_{model_name_lower}"}},
            {{"name": "publish", "title": "Publish", "icon": "globe", "permission": "publish_{model_name_lower}"}},
            {{"name": "archive", "title": "Archive", "icon": "archive", "permission": "archive_{model_name_lower}"}}
        ]
    }},
    
    # Analysis component for AI insights
    {{
        "type": "component",
        "path": "components/{model_name_lower}/{model_name}Analyze.tsx",
        "model": "{model_name}",
        "advanced": True,
        "features": [
            "ai_insights",          # AI-generated insights
            "trend_analysis",       # Trend analysis
            "competitive_analysis", # Competitive analysis
            "performance_metrics",  # Key performance indicators
            "market_analysis",      # Market analysis
            "recommendations",      # Actionable recommendations
            "interactive_charts",   # Interactive visualizations
            "custom_queries",       # Custom analysis queries
            "export_reports",       # Export analysis reports
            "scheduled_reports"     # Schedule recurring reports
        ],
        "analysis_types": [
            {{"name": "basic", "title": "Basic Analysis", "description": "Quick overview of key metrics"}},
            {{"name": "detailed", "title": "Detailed Analysis", "description": "In-depth analysis of all aspects"}},
            {{"name": "competitive", "title": "Competitive Analysis", "description": "Compare with similar products"}},
            {{"name": "predictive", "title": "Predictive Analysis", "description": "Forecast future performance"}},
            {{"name": "custom", "title": "Custom Analysis", "description": "Define your own analysis parameters"}}
        ]
    }},
    
    # Statistics component with charts and metrics
    {{
        "type": "component",
        "path": "components/{model_name_lower}/{model_name}Statistics.tsx",
        "model": "{model_name}",
        "advanced": True,
        "features": [
            "charts",               # Various chart types
            "metrics",              # Key metrics
            "trends",               # Trend analysis
            "comparisons",          # Comparative analysis
            "filtering",            # Filter data for analysis
            "time_periods",         # Different time periods
            "export_data",          # Export statistics
            "dashboards",           # Custom dashboards
            "alerts"                # Threshold-based alerts
        ],
        "chart_types": [
            {{"name": "sales", "title": "Sales Over Time", "type": "line", "data_source": "sales_by_period"}},
            {{"name": "categories", "title": "Distribution by Category", "type": "pie", "data_source": "count_by_category"}},
            {{"name": "inventory", "title": "Inventory Levels", "type": "bar", "data_source": "inventory_by_product"}},
            {{"name": "price", "title": "Price Distribution", "type": "histogram", "data_source": "price_distribution"}}
        ],
        "metrics": [
            {{"name": "total_sales", "title": "Total Sales", "format": "currency"}},
            {{"name": "average_rating", "title": "Average Rating", "format": "number"}},
            {{"name": "stock_level", "title": "Stock Level", "format": "number"}},
            {{"name": "growth_rate", "title": "Growth Rate", "format": "percent"}}
        ]
    }}
]

# Define the test files to be created with comprehensive test coverage
tests = [
    # Main model tests - comprehensive test suite
    {{
        "type": "comprehensive",
        "path": "tests/{model_name_lower}/test_{model_name_lower}.py",
        "model": "{model_name}",
        "controller": "{model_name_lower}_controller",
        "service": "{model_name}Service",
        "test_categories": [
            "model_validation",     # Model validation tests
            "service_crud",         # Service CRUD operation tests
            "service_advanced",     # Service advanced operation tests
            "controller_crud",      # Controller CRUD endpoint tests
            "controller_advanced",  # Controller advanced endpoint tests
            "authentication",       # Authentication tests
            "authorization",        # Authorization tests
            "performance",          # Performance tests
            "integration",          # Integration tests
            "edge_cases"            # Edge case tests
        ],
        "mocks": [
            "database",             # Database mocks
            "cache",                # Cache mocks
            "authentication",       # Authentication mocks
            "external_services",    # External service mocks
            "mcp_client"            # MCP client mocks
        ],
        "test_cases": [
            # Model tests
            {{"name": "test_create_minimum", "category": "model_validation", "description": "Test creation with minimum required fields"}},
            {{"name": "test_create_complete", "category": "model_validation", "description": "Test creation with all fields"}},
            {{"name": "test_validation_price", "category": "model_validation", "description": "Test price validation rules"}},
            {{"name": "test_validation_code", "category": "model_validation", "description": "Test code format validation"}},
            {{"name": "test_computed_properties", "category": "model_validation", "description": "Test computed properties"}},
            
            # Service tests
            {{"name": "test_service_create", "category": "service_crud", "description": "Test service create method"}},
            {{"name": "test_service_get", "category": "service_crud", "description": "Test service get method"}},
            {{"name": "test_service_update", "category": "service_crud", "description": "Test service update method"}},
            {{"name": "test_service_delete", "category": "service_crud", "description": "Test service delete method"}},
            {{"name": "test_service_list", "category": "service_crud", "description": "Test service list method"}},
            {{"name": "test_service_filtering", "category": "service_advanced", "description": "Test service filtering"}},
            {{"name": "test_service_sorting", "category": "service_advanced", "description": "Test service sorting"}},
            {{"name": "test_service_pagination", "category": "service_advanced", "description": "Test service pagination"}},
            {{"name": "test_service_caching", "category": "service_advanced", "description": "Test service caching"}},
            {{"name": "test_service_transactions", "category": "service_advanced", "description": "Test transaction handling"}},
            
            # Controller tests
            {{"name": "test_endpoint_create", "category": "controller_crud", "description": "Test create endpoint"}},
            {{"name": "test_endpoint_get", "category": "controller_crud", "description": "Test get endpoint"}},
            {{"name": "test_endpoint_update", "category": "controller_crud", "description": "Test update endpoint"}},
            {{"name": "test_endpoint_delete", "category": "controller_crud", "description": "Test delete endpoint"}},
            {{"name": "test_endpoint_list", "category": "controller_crud", "description": "Test list endpoint"}},
            {{"name": "test_auth_required", "category": "authentication", "description": "Test authentication requirements"}},
            {{"name": "test_permission_admin", "category": "authorization", "description": "Test admin permissions"}},
            {{"name": "test_permission_editor", "category": "authorization", "description": "Test editor permissions"}},
            {{"name": "test_permission_viewer", "category": "authorization", "description": "Test viewer permissions"}},
            
            # Advanced tests
            {{"name": "test_bulk_operations", "category": "controller_advanced", "description": "Test bulk operations"}},
            {{"name": "test_export_formats", "category": "controller_advanced", "description": "Test export formats"}},
            {{"name": "test_advanced_filtering", "category": "controller_advanced", "description": "Test advanced filtering"}},
            {{"name": "test_edge_case_invalid_input", "category": "edge_cases", "description": "Test invalid input handling"}},
            {{"name": "test_edge_case_very_large_input", "category": "edge_cases", "description": "Test very large input handling"}},
            {{"name": "test_performance_large_list", "category": "performance", "description": "Test performance with large lists"}}
        ]
    }},
    
    # UI component tests
    {{
        "type": "ui_component",
        "path": "tests/ui/{model_name_lower}/test_{model_name_lower}_components.tsx",
        "components": ["{model_name}List", "{model_name}Form", "{model_name}Detail", "{model_name}Analyze"],
        "test_frameworks": ["jest", "react-testing-library", "cypress"],
        "test_categories": [
            "rendering",            # Rendering tests
            "interaction",          # User interaction tests
            "state_management",     # State management tests
            "api_integration",      # API integration tests
            "error_handling",       # Error handling tests
            "accessibility"         # Accessibility tests
        ]
    }},
    
    # Integration tests
    {{
        "type": "integration",
        "path": "tests/integration/test_{model_name_lower}_workflow.py",
        "description": "End-to-end tests for {model_name} workflows",
        "test_workflows": [
            "create_update_delete",
            "search_filter_export",
            "bulk_operations",
            "role_based_access"
        ]
    }},
    
    # Performance tests
    {{
        "type": "performance",
        "path": "tests/performance/test_{model_name_lower}_performance.py",
        "description": "Performance tests for {model_name} operations",
        "test_scenarios": [
            "list_large_dataset",
            "complex_filtering",
            "concurrent_operations",
            "caching_effectiveness"
        ]
    }}
]

# Define the deployment configurations with detailed options for different environments
deployment = [
    # Docker Compose for local development
    {{
        "type": "docker",
        "path": "docker-compose.yml",
        "model": "{model_name}",
        "service": "{model_name}Service",
        "services": [
            # Core application services
            {{"name": "api", "image": "{model_name_lower}-api", "port": 9997, "depends_on": ["db", "redis", "mcp"]}},
            {{"name": "ui", "image": "{model_name_lower}-ui", "port": 9996, "depends_on": ["api"]}},
            
            # Database services
            {{"name": "db", "image": "postgres:14-alpine", "port": 5432, "volumes": ["{model_name_lower}-db-data"]}},
            {{"name": "redis", "image": "redis:alpine", "port": 6379, "volumes": ["{model_name_lower}-redis-data"]}},
            
            # Supporting services
            {{"name": "mcp", "image": "{model_name_lower}-mcp", "port": 8888, "depends_on": ["api"]}},
            {{"name": "search", "image": "opensearch:latest", "port": 9200, "volumes": ["{model_name_lower}-search-data"]}},
            {{"name": "analytics", "image": "grafana/grafana", "port": 3000, "volumes": ["{model_name_lower}-analytics-data"]}},
            
            # Infrastructure services
            {{"name": "prometheus", "image": "prom/prometheus", "port": 9090, "volumes": ["{model_name_lower}-prometheus-data"]}},
            {{"name": "jaeger", "image": "jaegertracing/all-in-one", "port": 16686}},
            {{"name": "mailhog", "image": "mailhog/mailhog", "ports": [1025, 8025]}}
        ],
        "networks": ["{model_name_lower}-network"],
        "volumes": [
            "{model_name_lower}-db-data",
            "{model_name_lower}-redis-data",
            "{model_name_lower}-search-data",
            "{model_name_lower}-analytics-data",
            "{model_name_lower}-prometheus-data"
        ],
        "environment_variables": [
            "DATABASE_URL=postgresql://{{{{DB_USER}}}}:{{{{DB_PASSWORD}}}}@db:5432/{{{{DB_NAME}}}}",
            "REDIS_URL=redis://redis:6379/0",
            "MCP_URL=http://mcp:8888/mcp",
            "SEARCH_URL=http://search:9200",
            "SECRET_KEY={{{{SECRET_KEY}}}}",
            "ENVIRONMENT=development",
            "LOG_LEVEL=debug"
        ]
    }},
    
    # Kubernetes configuration for production deployment
    {{
        "type": "kubernetes",
        "path": "kubernetes",
        "model": "{model_name}",
        "components": [
            # Core deployments
            {{"type": "deployment", "name": "api", "replicas": 3, "container": "{model_name_lower}-api", "resources": {{"cpu": "500m", "memory": "512Mi"}}}},
            {{"type": "deployment", "name": "ui", "replicas": 2, "container": "{model_name_lower}-ui", "resources": {{"cpu": "200m", "memory": "256Mi"}}}},
            {{"type": "deployment", "name": "mcp", "replicas": 2, "container": "{model_name_lower}-mcp", "resources": {{"cpu": "500m", "memory": "512Mi"}}}},
            
            # StatefulSets for databases
            {{"type": "statefulset", "name": "db", "replicas": 1, "container": "postgres:14", "volume": "{model_name_lower}-db"}},
            {{"type": "statefulset", "name": "redis", "replicas": 1, "container": "redis:alpine", "volume": "{model_name_lower}-redis"}},
            
            # Services
            {{"type": "service", "name": "api", "port": 9997, "target_port": 9997, "type": "ClusterIP"}},
            {{"type": "service", "name": "ui", "port": 9996, "target_port": 9996, "type": "ClusterIP"}},
            {{"type": "service", "name": "db", "port": 5432, "target_port": 5432, "type": "ClusterIP"}},
            {{"type": "service", "name": "redis", "port": 6379, "target_port": 6379, "type": "ClusterIP"}},
            {{"type": "service", "name": "mcp", "port": 8888, "target_port": 8888, "type": "ClusterIP"}},
            
            # Ingress
            {{"type": "ingress", "name": "{model_name_lower}", "hosts": ["{model_name_lower}.example.com"], "paths": ["/", "/api", "/mcp"]}},
            
            # ConfigMaps and Secrets
            {{"type": "configmap", "name": "{model_name_lower}-config", "data": {{"environment": "production", "log_level": "info"}}}},
            {{"type": "secret", "name": "{model_name_lower}-secrets", "data": {{"db-password": "...", "secret-key": "..."}}}},
            
            # Volumes
            {{"type": "persistentvolumeclaim", "name": "{model_name_lower}-db", "storage": "10Gi", "access_modes": ["ReadWriteOnce"]}},
            {{"type": "persistentvolumeclaim", "name": "{model_name_lower}-redis", "storage": "5Gi", "access_modes": ["ReadWriteOnce"]}}
        ],
        "namespaces": [
            {{"name": "{model_name_lower}-prod", "description": "Production environment"}},
            {{"name": "{model_name_lower}-staging", "description": "Staging environment"}},
            {{"name": "{model_name_lower}-dev", "description": "Development environment"}}
        ]
    }},
    
    # API Server Dockerfile with multi-stage build
    {{
        "type": "dockerfile",
        "path": "Dockerfile",
        "model": "{model_name}",
        "stages": [
            {{"name": "build", "base": "python:3.11-slim", "description": "Build stage for dependencies"}},
            {{"name": "runtime", "base": "python:3.11-slim", "description": "Runtime stage with minimal dependencies"}}
        ],
        "instructions": [
            # Build stage
            {{"stage": "build", "type": "workdir", "value": "/app"}},
            {{"stage": "build", "type": "copy", "src": "requirements.txt", "dest": "./"}},
            {{"stage": "build", "type": "run", "value": "pip install --no-cache-dir -r requirements.txt"}},
            
            # Runtime stage
            {{"stage": "runtime", "type": "workdir", "value": "/app"}},
            {{"stage": "runtime", "type": "copy", "src": ".", "dest": "."}},
            {{"stage": "runtime", "type": "copy", "from": "build", "src": "/usr/local/lib/python3.11/site-packages", "dest": "/usr/local/lib/python3.11/site-packages"}},
            {{"stage": "runtime", "type": "env", "key": "PYTHONPATH", "value": "/app"}},
            {{"stage": "runtime", "type": "env", "key": "PORT", "value": "9997"}},
            {{"stage": "runtime", "type": "expose", "port": "9997"}},
            {{"stage": "runtime", "type": "healthcheck", "cmd": "curl --fail http://localhost:9997/health || exit 1", "interval": "30s", "timeout": "10s", "retries": 3}},
            {{"stage": "runtime", "type": "cmd", "value": ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9997"]}}
        ]
    }},
    
    # UI Dockerfile with Node.js
    {{
        "type": "dockerfile",
        "path": "ui/Dockerfile",
        "model": "{model_name}",
        "ui": True,
        "stages": [
            {{"name": "build", "base": "node:18-alpine", "description": "Build stage for dependencies and assets"}},
            {{"name": "runtime", "base": "node:18-alpine", "description": "Runtime stage with built application"}}
        ],
        "instructions": [
            # Build stage
            {{"stage": "build", "type": "workdir", "value": "/app"}},
            {{"stage": "build", "type": "copy", "src": "package*.json", "dest": "./"}},
            {{"stage": "build", "type": "run", "value": "npm ci"}},
            {{"stage": "build", "type": "copy", "src": ".", "dest": "."}},
            {{"stage": "build", "type": "run", "value": "npm run build"}},
            
            # Runtime stage
            {{"stage": "runtime", "type": "workdir", "value": "/app"}},
            {{"stage": "runtime", "type": "copy", "src": "package*.json", "dest": "./"}},
            {{"stage": "runtime", "type": "run", "value": "npm ci --only=production"}},
            {{"stage": "runtime", "type": "copy", "from": "build", "src": "/app/.next", "dest": "/app/.next"}},
            {{"stage": "runtime", "type": "copy", "from": "build", "src": "/app/public", "dest": "/app/public"}},
            {{"stage": "runtime", "type": "env", "key": "PORT", "value": "9996"}},
            {{"stage": "runtime", "type": "env", "key": "NODE_ENV", "value": "production"}},
            {{"stage": "runtime", "type": "expose", "port": "9996"}},
            {{"stage": "runtime", "type": "healthcheck", "cmd": "wget --no-verbose --tries=1 --spider http://localhost:9996/health || exit 1", "interval": "30s", "timeout": "10s", "retries": 3}},
            {{"stage": "runtime", "type": "cmd", "value": ["npm", "run", "start"]}}
        ]
    }},
    
    # MCP Server Dockerfile
    {{
        "type": "dockerfile",
        "path": "mcp/Dockerfile",
        "model": "{model_name}",
        "description": "Docker image for the MCP server",
        "base_image": "python:3.11-slim",
        "expose_port": 8888,
        "environment_variables": [
            {{"name": "MCP_PORT", "value": "8888"}},
            {{"name": "API_URL", "value": "http://api:9997"}},
            {{"name": "LOG_LEVEL", "value": "info"}}
        ]
    }}
]

# MCP Integration with detailed component configuration
mcp_components = [
    # MCP Server definition
    {{
        "type": "server",
        "name": "{model_name}Server",
        "model": "{model_name}",
        "port": 8888,
        "description": "Model Context Protocol server for {model_name}",
        "capabilities": [
            "resources",        # Resource capabilities
            "tools",            # Tool capabilities
            "prompts",          # Prompt capabilities
            "streaming",        # Streaming capabilities
            "sampling"          # Sampling capabilities
        ],
        "security": [
            "api_key",          # API key authentication
            "jwt",              # JWT authentication
            "rate_limiting"     # Rate limiting
        ]
    }},
    
    # Resource definitions for accessing data
    {{
        "type": "resource",
        "name": "{model_name_lower}",
        "model": "{model_name}",
        "schema": "{model_name_lower}://{{{id}}}",
        "description": "Get a {model_name} by ID",
        "parameters": [
            {{"name": "id", "type": "string", "description": "The ID of the {model_name} to retrieve"}}
        ],
        "response_type": "application/json",
        "cacheable": True,
        "required_permissions": ["read_{model_name_lower}"]
    }},
    {{
        "type": "resource",
        "name": "{model_name_lower}_list",
        "model": "{model_name}",
        "schema": "{model_name_lower}://list?category={{{category}}}",
        "description": "Get a list of {model_name}s, optionally filtered by category",
        "parameters": [
            {{"name": "category", "type": "string", "description": "Optional category to filter by", "required": False}}
        ],
        "response_type": "application/json",
        "cacheable": True,
        "required_permissions": ["read_{model_name_lower}"]
    }},
    {{
        "type": "resource",
        "name": "{model_name_lower}_schema",
        "model": "{model_name}",
        "schema": "{model_name_lower}://schema",
        "description": "Get the JSON schema for {model_name}",
        "parameters": [],
        "response_type": "application/json",
        "cacheable": True,
        "required_permissions": []
    }},
    {{
        "type": "resource",
        "name": "{model_name_lower}_statistics",
        "model": "{model_name}",
        "schema": "{model_name_lower}://statistics",
        "description": "Get statistics about {model_name}s",
        "parameters": [],
        "response_type": "application/json",
        "cacheable": True,
        "required_permissions": ["read_statistics"]
    }},
    
    # Tool definitions for actions
    {{
        "type": "tool",
        "name": "create_{model_name_lower}",
        "model": "{model_name}",
        "operation": "create",
        "description": "Create a new {model_name}",
        "parameters": [
            {{"name": "name", "type": "string", "description": "Name of the {model_name}", "required": True}},
            {{"name": "description", "type": "string", "description": "Description of the {model_name}", "required": False}},
            {{"name": "price", "type": "number", "description": "Price of the {model_name}", "required": True}},
            {{"name": "category", "type": "string", "description": "Category of the {model_name}", "required": False}}
        ],
        "response_type": "application/json",
        "required_permissions": ["create_{model_name_lower}"]
    }},
    {{
        "type": "tool",
        "name": "update_{model_name_lower}",
        "model": "{model_name}",
        "operation": "update",
        "description": "Update an existing {model_name}",
        "parameters": [
            {{"name": "id", "type": "string", "description": "ID of the {model_name} to update", "required": True}},
            {{"name": "name", "type": "string", "description": "New name of the {model_name}", "required": False}},
            {{"name": "description", "type": "string", "description": "New description of the {model_name}", "required": False}},
            {{"name": "price", "type": "number", "description": "New price of the {model_name}", "required": False}},
            {{"name": "category", "type": "string", "description": "New category of the {model_name}", "required": False}}
        ],
        "response_type": "application/json",
        "required_permissions": ["update_{model_name_lower}"]
    }},
    {{
        "type": "tool",
        "name": "delete_{model_name_lower}",
        "model": "{model_name}",
        "operation": "delete",
        "description": "Delete a {model_name}",
        "parameters": [
            {{"name": "id", "type": "string", "description": "ID of the {model_name} to delete", "required": True}}
        ],
        "response_type": "application/json",
        "required_permissions": ["delete_{model_name_lower}"]
    }},
    {{
        "type": "tool",
        "name": "analyze_{model_name_lower}",
        "model": "{model_name}",
        "operation": "analyze",
        "description": "Analyze a {model_name} with LLM",
        "parameters": [
            {{"name": "id", "type": "string", "description": "ID of the {model_name} to analyze", "required": True}},
            {{"name": "question", "type": "string", "description": "Question to ask about the {model_name}", "required": True}}
        ],
        "response_type": "text/plain",
        "required_permissions": ["analyze_{model_name_lower}"]
    }},
    {{
        "type": "tool",
        "name": "search_{model_name_lower}s",
        "model": "{model_name}",
        "operation": "search",
        "description": "Search for {model_name}s",
        "parameters": [
            {{"name": "query", "type": "string", "description": "Search query", "required": True}},
            {{"name": "fields", "type": "array", "description": "Fields to search in", "required": False}},
            {{"name": "limit", "type": "integer", "description": "Maximum number of results to return", "required": False}}
        ],
        "response_type": "application/json",
        "required_permissions": ["read_{model_name_lower}"]
    }},
    {{
        "type": "tool",
        "name": "recommend_{model_name_lower}s",
        "model": "{model_name}",
        "operation": "recommend",
        "description": "Get recommendations for similar {model_name}s",
        "parameters": [
            {{"name": "id", "type": "string", "description": "ID of the reference {model_name}", "required": True}},
            {{"name": "limit", "type": "integer", "description": "Maximum number of recommendations to return", "required": False}}
        ],
        "response_type": "application/json",
        "required_permissions": ["read_{model_name_lower}"]
    }},
    
    # Prompt templates
    {{
        "type": "prompt",
        "name": "summarize_{model_name_lower}",
        "description": "Generate a summary of a {model_name}",
        "parameters": [
            {{"name": "id", "type": "string", "description": "ID of the {model_name} to summarize", "required": True}}
        ],
        "template": "Please provide a concise summary of this {model_name}:\n\n{{{{product_data}}}}"
    }},
    {{
        "type": "prompt",
        "name": "compare_{model_name_lower}s",
        "description": "Compare two {model_name}s",
        "parameters": [
            {{"name": "id1", "type": "string", "description": "ID of the first {model_name}", "required": True}},
            {{"name": "id2", "type": "string", "description": "ID of the second {model_name}", "required": True}}
        ],
        "template": "Please compare these two {model_name}s:\n\nProduct 1:\n{{{{product1_data}}}}\n\nProduct 2:\n{{{{product2_data}}}}\n\nProvide a detailed comparison of their features, price, and value."
    }}
]

# Authentication configuration with detailed role-based access control
auth = {{
    "roles": [
        {{"name": "admin", "description": "Full access to all features"}},
        {{"name": "editor", "description": "Can create and edit but not delete"}},
        {{"name": "viewer", "description": "Read-only access"}},
        {{"name": "analyst", "description": "Can view and analyze data"}},
        {{"name": "manager", "description": "Can manage but not edit technical details"}}
    ],
    "permissions": {{
        # Basic CRUD permissions
        "create_{model_name_lower}": ["admin", "editor"],
        "read_{model_name_lower}": ["admin", "editor", "viewer", "analyst", "manager"],
        "update_{model_name_lower}": ["admin", "editor", "manager"],
        "delete_{model_name_lower}": ["admin"],
        
        # List and search permissions
        "list_{model_name_lower}s": ["admin", "editor", "viewer", "analyst", "manager"],
        "search_{model_name_lower}s": ["admin", "editor", "viewer", "analyst", "manager"],
        "filter_{model_name_lower}s": ["admin", "editor", "viewer", "analyst", "manager"],
        
        # Advanced permissions
        "analyze_{model_name_lower}": ["admin", "editor", "analyst"],
        "export_{model_name_lower}s": ["admin", "editor", "analyst", "manager"],
        "bulk_create_{model_name_lower}s": ["admin"],
        "bulk_update_{model_name_lower}s": ["admin"],
        
        # Workflow permissions
        "publish_{model_name_lower}": ["admin", "editor", "manager"],
        "unpublish_{model_name_lower}": ["admin", "editor", "manager"],
        "archive_{model_name_lower}": ["admin", "manager"],
        
        # Statistics and reporting
        "view_statistics": ["admin", "analyst", "manager"],
        "generate_reports": ["admin", "analyst", "manager"],
        
        # System-level permissions
        "manage_users": ["admin"],
        "manage_permissions": ["admin"],
        "view_audit_logs": ["admin"],
        "configure_system": ["admin"]
    }},
    "authentication_methods": [
        {{"type": "jwt", "description": "JWT token-based authentication"}},
        {{"type": "oauth2", "description": "OAuth2 authentication with providers"}},
        {{"type": "api_key", "description": "API key authentication for services"}}
    ],
    "oauth_providers": [
        {{"name": "google", "client_id": "${{GOOGLE_CLIENT_ID}}", "client_secret": "${{GOOGLE_CLIENT_SECRET}}"}},
        {{"name": "github", "client_id": "${{GITHUB_CLIENT_ID}}", "client_secret": "${{GITHUB_CLIENT_SECRET}}"}},
        {{"name": "microsoft", "client_id": "${{MICROSOFT_CLIENT_ID}}", "client_secret": "${{MICROSOFT_CLIENT_SECRET}}"}}
    ],
    "jwt_config": {{
        "secret_key": "${{JWT_SECRET_KEY}}",
        "algorithm": "HS256",
        "access_token_expire_minutes": 30,
        "refresh_token_expire_days": 7
    }}
}}

# Cache configuration with detailed caching strategy
cache = {{
    "ttl": 300,  # Default TTL: 5 minutes
    "categories": [
        {{"name": "entity", "ttl": 300, "description": "Entity caching for individual records"}},
        {{"name": "list", "ttl": 60, "description": "List caching for collections"}},
        {{"name": "computed", "ttl": 600, "description": "Caching for computed values"}},
        {{"name": "statistics", "ttl": 1800, "description": "Caching for statistics"}},
        {{"name": "user", "ttl": 1800, "description": "User-specific data caching"}}
    ],
    "keys": [
        # Entity caches
        {{"pattern": "{model_name_lower}_{{id}}", "category": "entity", "description": "Individual {model_name} cache"}},
        {{"pattern": "category_{{id}}", "category": "entity", "description": "Individual Category cache"}},
        
        # List caches
        {{"pattern": "all_{model_name_lower}s", "category": "list", "description": "All {model_name}s list cache"}},
        {{"pattern": "{model_name_lower}s_by_category_{{category}}", "category": "list", "description": "{model_name}s filtered by category"}},
        {{"pattern": "{model_name_lower}s_by_status_{{status}}", "category": "list", "description": "{model_name}s filtered by status"}},
        {{"pattern": "{model_name_lower}s_search_{{query}}", "category": "list", "description": "{model_name}s search results"}},
        
        # Computed caches
        {{"pattern": "{model_name_lower}_{{id}}_related", "category": "computed", "description": "Related {model_name}s"}},
        {{"pattern": "{model_name_lower}_{{id}}_recommendations", "category": "computed", "description": "Recommended {model_name}s"}},
        
        # Statistics caches
        {{"pattern": "{model_name_lower}_stats", "category": "statistics", "description": "Overall {model_name} statistics"}},
        {{"pattern": "{model_name_lower}_stats_by_category", "category": "statistics", "description": "{model_name} statistics by category"}},
        
        # User-specific caches
        {{"pattern": "user_{{user_id}}_{model_name_lower}s", "category": "user", "description": "User's {model_name}s"}},
        {{"pattern": "user_{{user_id}}_recent_{model_name_lower}s", "category": "user", "description": "User's recently viewed {model_name}s"}}
    ],
    "invalidation_rules": [
        {{"event": "create_{model_name_lower}", "invalidate": ["all_{model_name_lower}s", "{model_name_lower}s_by_category_*", "{model_name_lower}_stats"]}},
        {{"event": "update_{model_name_lower}", "invalidate": ["{model_name_lower}_{{id}}", "all_{model_name_lower}s", "{model_name_lower}s_by_category_*", "{model_name_lower}_stats"]}},
        {{"event": "delete_{model_name_lower}", "invalidate": ["{model_name_lower}_{{id}}", "all_{model_name_lower}s", "{model_name_lower}s_by_category_*", "{model_name_lower}_stats"]}}
    ],
    "storage_options": [
        {{"type": "redis", "description": "Redis cache for distributed environments", "config": {{"url": "redis://redis:6379/0", "password": "${{REDIS_PASSWORD}}"}}}},
        {{"type": "memory", "description": "In-memory cache for single-instance deployments", "config": {{"max_size": 1000}}}},
        {{"type": "memcached", "description": "Memcached for high-performance caching", "config": {{"servers": ["memcached:11211"]}}}}
    ]
}}

# Database configuration with detailed schema and migration settings
database = {{
    "engine": "postgres",
    "connection": {{
        "host": "${{DB_HOST}}",
        "port": "${{DB_PORT}}",
        "database": "${{DB_NAME}}",
        "user": "${{DB_USER}}",
        "password": "${{DB_PASSWORD}}",
        "ssl": True,
        "pool_size": 10,
        "max_overflow": 20,
        "timeout": 30
    }},
    "migrations": {{
        "enabled": True,
        "auto_migrate": True,
        "migration_dir": "migrations",
        "version_table": "alembic_version",
        "naming_convention": {{
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s"
        }}
    }},
    "tables": [
        {{
            "name": "{model_name_lower}s",
            "description": "Stores {model_name} records",
            "schema": "public",
            "columns": [
                {{"name": "id", "type": "uuid", "primary_key": True, "default": "uuid_generate_v4()"}},
                {{"name": "name", "type": "varchar(255)", "nullable": False}},
                {{"name": "description", "type": "text", "nullable": True}},
                {{"name": "code", "type": "varchar(8)", "nullable": False}},
                {{"name": "price", "type": "decimal(10,2)", "nullable": False}},
                {{"name": "discount_percent", "type": "decimal(5,2)", "nullable": False, "default": 0}},
                {{"name": "quantity", "type": "integer", "nullable": False, "default": 0}},
                {{"name": "category", "type": "varchar(100)", "nullable": True}},
                {{"name": "status", "type": "varchar(20)", "nullable": False, "default": "'active'"}},
                {{"name": "owner_id", "type": "uuid", "nullable": False}},
                {{"name": "created_at", "type": "timestamp with time zone", "nullable": False, "default": "CURRENT_TIMESTAMP"}},
                {{"name": "updated_at", "type": "timestamp with time zone", "nullable": False, "default": "CURRENT_TIMESTAMP"}}
            ]
        }},
        {{
            "name": "categories",
            "description": "Stores category records",
            "schema": "public",
            "columns": [
                {{"name": "id", "type": "uuid", "primary_key": True, "default": "uuid_generate_v4()"}},
                {{"name": "name", "type": "varchar(100)", "nullable": False}},
                {{"name": "slug", "type": "varchar(100)", "nullable": False}},
                {{"name": "description", "type": "text", "nullable": True}},
                {{"name": "parent_id", "type": "uuid", "nullable": True}},
                {{"name": "created_at", "type": "timestamp with time zone", "nullable": False, "default": "CURRENT_TIMESTAMP"}},
                {{"name": "updated_at", "type": "timestamp with time zone", "nullable": False, "default": "CURRENT_TIMESTAMP"}}
            ]
        }},
        {{
            "name": "reviews",
            "description": "Stores product reviews",
            "schema": "public",
            "columns": [
                {{"name": "id", "type": "uuid", "primary_key": True, "default": "uuid_generate_v4()"}},
                {{"name": "product_id", "type": "uuid", "nullable": False}},
                {{"name": "user_id", "type": "uuid", "nullable": False}},
                {{"name": "title", "type": "varchar(255)", "nullable": False}},
                {{"name": "content", "type": "text", "nullable": False}},
                {{"name": "rating", "type": "integer", "nullable": False}},
                {{"name": "created_at", "type": "timestamp with time zone", "nullable": False, "default": "CURRENT_TIMESTAMP"}},
                {{"name": "updated_at", "type": "timestamp with time zone", "nullable": False, "default": "CURRENT_TIMESTAMP"}}
            ]
        }}
    ],
    "indexes": [
        {{"table": "{model_name_lower}s", "columns": ["name"], "unique": False, "type": "btree"}},
        {{"table": "{model_name_lower}s", "columns": ["code"], "unique": True, "type": "btree"}},
        {{"table": "{model_name_lower}s", "columns": ["category"], "unique": False, "type": "btree"}},
        {{"table": "{model_name_lower}s", "columns": ["created_at"], "unique": False, "type": "btree"}},
        {{"table": "{model_name_lower}s", "columns": ["owner_id"], "unique": False, "type": "btree"}},
        {{"table": "{model_name_lower}s", "columns": ["status"], "unique": False, "type": "btree"}},
        {{"table": "{model_name_lower}s", "columns": ["price"], "unique": False, "type": "btree"}},
        {{"table": "categories", "columns": ["slug"], "unique": True, "type": "btree"}},
        {{"table": "categories", "columns": ["parent_id"], "unique": False, "type": "btree"}},
        {{"table": "reviews", "columns": ["product_id"], "unique": False, "type": "btree"}},
        {{"table": "reviews", "columns": ["user_id"], "unique": False, "type": "btree"}},
        {{"table": "reviews", "columns": ["rating"], "unique": False, "type": "btree"}}
    ],
    "constraints": [
        {{"type": "unique", "table": "{model_name_lower}s", "columns": ["code"]}},
        {{"type": "unique", "table": "{model_name_lower}s", "columns": ["name", "owner_id"]}},
        {{"type": "check", "table": "{model_name_lower}s", "condition": "quantity >= 0"}},
        {{"type": "check", "table": "{model_name_lower}s", "condition": "price >= 0"}},
        {{"type": "check", "table": "{model_name_lower}s", "condition": "discount_percent >= 0 AND discount_percent <= 100"}},
        {{"type": "check", "table": "reviews", "condition": "rating >= 1 AND rating <= 5"}},
        {{"type": "foreign key", "table": "reviews", "columns": ["product_id"], "references_table": "{model_name_lower}s", "references_columns": ["id"], "on_delete": "CASCADE"}},
        {{"type": "foreign key", "table": "categories", "columns": ["parent_id"], "references_table": "categories", "references_columns": ["id"], "on_delete": "SET NULL"}}
    ],
    "search_configuration": {{
        "engine": "postgresql_full_text",
        "indexes": [
            {{"table": "{model_name_lower}s", "columns": ["name", "description"], "weights": {{"name": "A", "description": "B"}}}},
            {{"table": "categories", "columns": ["name", "description"]}},
            {{"table": "reviews", "columns": ["title", "content"]}}
        ]
    }}
}}

# API documentation configuration
api_docs = {{
    "title": "{model_name} API",
    "description": "API for managing {model_name}s",
    "version": "1.0.0",
    "contact": {{
        "name": "API Support",
        "email": "api-support@example.com",
        "url": "https://example.com/support"
    }},
    "license": {{
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }},
    "servers": [
        {{"url": "https://api.example.com/v1", "description": "Production server"}},
        {{"url": "https://staging-api.example.com/v1", "description": "Staging server"}},
        {{"url": "http://localhost:9997", "description": "Local development server"}}
    ],
    "tags": [
        {{"name": "{model_name_lower}", "description": "{model_name} operations"}},
        {{"name": "category", "description": "Category operations"}},
        {{"name": "review", "description": "Review operations"}},
        {{"name": "statistics", "description": "Statistical operations"}}
    ],
    "security_schemes": [
        {{"name": "bearer_auth", "type": "http", "scheme": "bearer", "bearer_format": "JWT"}},
        {{"name": "api_key", "type": "apiKey", "in": "header", "name": "X-API-Key"}}
    ]
}}
"""
    
    # Write the file
    with open(output_path, "w") as f:
        f.write(content)
    
    return output_path

def create_advanced_scaffold_from_config(config_path: str, output_dir: str = None) -> Dict[str, List[str]]:
    """Create an advanced scaffold from a configuration file
    
    Args:
        config_path: Path to the scaffold config file
        output_dir: Directory to output the scaffold
        
    Returns:
        Dictionary of created files by category
    """
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} does not exist")
        sys.exit(1)
    
    # Load the config
    spec = importlib.util.spec_from_file_location("scaffold_config", config_path)
    if spec and spec.loader:
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
    else:
        print(f"Error: Could not load config from {config_path}")
        sys.exit(1)
    
    if output_dir is None:
        output_dir = os.getcwd()
    
    # Create the directories
    created_files = {
        "models": [],
        "controllers": [],
        "services": [],
        "ui": [],
        "tests": [],
        "deployment": [],
        "mcp": [],
    }
    
    # Ensure all necessary directories exist
    os.makedirs(os.path.join(output_dir, "backend", "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "backend", "controllers"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "backend", "services"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "backend", "cache"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "backend", "auth"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ui", "app", "dashboard"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ui", "components"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tests"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "backend", "mcp"), exist_ok=True)
    
    # Create models
    for model_def in getattr(config, "models", []):
        model_name = model_def["name"]
        model_file = f"{model_name.lower()}.py"
        model_path = os.path.join(output_dir, "backend", "models", model_file)
        
        # Format the fields
        fields_content = []
        for field in model_def.get("fields", []):
            field_name = field["name"]
            field_type = field["type"]
            required = field.get("required", True)
            default = field.get("default", None)
            
            if required and default is None:
                fields_content.append(f"    {field_name}: {field_type} = Field(..., description=\"{field_name.title()} of the {model_name}\")")
            else:
                if default is not None:
                    fields_content.append(f"    {field_name}: {field_type} = Field({default}, description=\"{field_name.title()} of the {model_name}\")")
                else:
                    fields_content.append(f"    {field_name}: Optional[{field_type}] = Field(None, description=\"{field_name.title()} of the {model_name}\")")
        
        # Format the relationships
        relations_content = []
        for relation in model_def.get("relations", []):
            relation_name = relation["name"]
            relation_type = relation["type"]
            required = relation.get("required", True)
            default = relation.get("default", None)
            docstring = relation.get("docstring", f"Relation to {relation_name}")
            
            if required and default is None:
                relations_content.append(f"    {relation_name}: {relation_type} = Field(..., description=\"{docstring}\")")
            else:
                if default is not None:
                    relations_content.append(f"    {relation_name}: {relation_type} = Field({default}, description=\"{docstring}\")")
                else:
                    relations_content.append(f"    {relation_name}: Optional[{relation_type}] = Field(None, description=\"{docstring}\")")
        
        # Create the model content
        model_content = ADVANCED_MODEL_TEMPLATE.format(
            model_name=model_name,
            docstring=model_def.get("docstring", f"{model_name} data model"),
            fields="\n".join(fields_content),
            relations="\n".join(relations_content)
        )
        
        # Write the model file
        with open(model_path, "w") as f:
            f.write(model_content)
        
        created_files["models"].append(model_path)
    
    # Create services
    for model_def in getattr(config, "models", []):
        model_name = model_def["name"]
        model_file = model_name.lower()
        service_name = f"{model_name}Service"
        service_file = f"{model_name.lower()}_service.py"
        service_path = os.path.join(output_dir, "backend", "services", service_file)
        
        model_name_lower = model_name.lower()
        
        # Create the service content
        service_content = ADVANCED_SERVICE_TEMPLATE.format(
            model_name=model_name,
            model_file=model_file,
            service_name=service_name,
            model_name_lower=model_name_lower
        )
        
        # Write the service file
        with open(service_path, "w") as f:
            f.write(service_content)
        
        created_files["services"].append(service_path)
    
    # Create controllers
    for endpoint_def in getattr(config, "endpoints", []):
        endpoint_name = endpoint_def["name"]
        model_name = endpoint_def["model"]
        model_file = model_name.lower()
        service_name = f"{model_name}Service"
        service_file = f"{model_name.lower()}_service"
        controller_file = f"{endpoint_name}_controller.py"
        controller_path = os.path.join(output_dir, "backend", "controllers", controller_file)
        
        # Create the controller content
        controller_content = ADVANCED_CONTROLLER_TEMPLATE.format(
            name=endpoint_name,
            model_name=model_name,
            model_file=model_file,
            service_name=service_name,
            service_file=service_file,
            route_prefix=endpoint_name,
            tag=endpoint_name,
            operation_name=endpoint_name,
            id_param="id"
        )
        
        # Write the controller file
        with open(controller_path, "w") as f:
            f.write(controller_content)
        
        created_files["controllers"].append(controller_path)
    
    # Create UI components
    for ui_def in getattr(config, "ui_components", []):
        component_type = ui_def["type"]
        path = ui_def["path"]
        model_name = ui_def.get("model")
        model_name_lower = model_name.lower()
        advanced = ui_def.get("advanced", False)
        
        full_path = os.path.join(output_dir, "ui", path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        if component_type == "component":
            if path.endswith("List.tsx") and advanced:
                # Advanced list component
                content = ADVANCED_UI_LIST_TEMPLATE.format(
                    model_name=model_name,
                    model_name_lower=model_name_lower
                )
            elif path.endswith("Form.tsx") and advanced:
                # Advanced form component
                content = ADVANCED_UI_FORM_TEMPLATE.format(
                    model_name=model_name,
                    model_name_lower=model_name_lower
                )
            else:
                # Simple component template (placeholder)
                content = f"""import React from 'react';

export default function {model_name}{path.split('/')[-1].split('.')[0]}() {{
  return (
    <div className="{model_name_lower}-component">
      <h2>{model_name} Component</h2>
      <p>Advanced component for {model_name}</p>
    </div>
  );
}}
"""
        else:
            # Page component (placeholder)
            content = f"""import React from 'react';
import {model_name}List from '../../components/{model_name_lower}/{model_name}List';
import {model_name}Form from '../../components/{model_name_lower}/{model_name}Form';

export default function {model_name}Page() {{
  return (
    <div className="{model_name_lower}-page">
      <h1>{model_name} Dashboard</h1>
      <div className="dashboard-content">
        <div className="list-section">
          <{model_name}List />
        </div>
      </div>
    </div>
  );
}}
"""
        
        # Write the UI file
        with open(full_path, "w") as f:
            f.write(content)
        
        created_files["ui"].append(full_path)
    
    # Create tests
    for test_def in getattr(config, "tests", []):
        test_type = test_def["type"]
        path = test_def["path"]
        model_name = test_def["model"]
        controller_file = test_def.get("controller", f"{model_name.lower()}_controller")
        service_name = test_def.get("service", f"{model_name}Service")
        
        full_path = os.path.join(output_dir, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        if test_type == "comprehensive":
            # Comprehensive test suite
            content = ADVANCED_TEST_TEMPLATE.format(
                model_name=model_name,
                model_file=model_name.lower(),
                service_name=service_name,
                service_file=f"{model_name.lower()}_service",
                controller_file=controller_file,
                route_prefix=model_name.lower(),
                model_name_lower=model_name.lower()
            )
        else:
            # Simple test template (placeholder)
            content = f"""import pytest
from fortitude.backend.models.{model_name.lower()} import {model_name}

def test_{model_name.lower()}():
    """Test {model_name}"""
    instance = {model_name}(name="Test {model_name}")
    assert instance.name == "Test {model_name}"
    assert instance.id is not None
"""
        
        # Write the test file
        with open(full_path, "w") as f:
            f.write(content)
        
        created_files["tests"].append(full_path)
    
    # Create deployment files
    for deploy_def in getattr(config, "deployment", []):
        deploy_type = deploy_def["type"]
        path = deploy_def["path"]
        model_name = deploy_def["model"]
        service_name = deploy_def.get("service", f"{model_name}Service")
        ui = deploy_def.get("ui", False)
        
        full_path = os.path.join(output_dir, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        if deploy_type == "docker":
            # Docker Compose file
            content = DEPLOYMENT_TEMPLATE.format(
                model_name=model_name,
                service_name=service_name,
                service_name_lower=model_name.lower()
            )
        elif deploy_type == "dockerfile" and not ui:
            # Backend Dockerfile
            content = DOCKERFILE_TEMPLATE.format(
                model_name=model_name
            )
        elif deploy_type == "dockerfile" and ui:
            # UI Dockerfile
            content = UI_DOCKERFILE_TEMPLATE.format(
                model_name=model_name
            )
        else:
            # Generic deployment file
            content = f"""# Deployment configuration for {model_name}
# Generated by Fortitude advanced scaffolding

name: {model_name}
type: {deploy_type}
"""
        
        # Write the deployment file
        with open(full_path, "w") as f:
            f.write(content)
        
        created_files["deployment"].append(full_path)
    
    # Create MCP components
    for mcp_def in getattr(config, "mcp_components", []):
        component_type = mcp_def["type"]
        name = mcp_def["name"]
        model_name = mcp_def["model"]
        
        if component_type == "server":
            # MCP Server
            server_path = os.path.join(output_dir, "backend", "mcp", f"{model_name.lower()}_server.py")
            server_content = f"""from fortitude.backend.mcp import FortitudeMCP
from ..models.{model_name.lower()} import {model_name}
from ..services.{model_name.lower()}_service import {model_name}Service

# Create an MCP server for {model_name}
mcp = FortitudeMCP("{model_name} MCP Server")
service = {model_name}Service()

@mcp.resource("{model_name.lower()}://{{id}}")
async def get_{model_name.lower()}(id: str) -> str:
    """Get a {model_name} by ID"""
    instance = await service.get(id)
    if instance:
        return instance.model_dump_json(indent=2)
    return f"{{{{ \\"error\\": \\"{model_name} not found\\" }}}}"

@mcp.tool()
async def create_{model_name.lower()}(name: str, description: str = None, **kwargs) -> str:
    """Create a new {model_name}"""
    data = {{
        "name": name,
        "description": description,
        **kwargs
    }}
    instance = await service.create(data)
    return instance.model_dump_json(indent=2)

@mcp.tool()
async def update_{model_name.lower()}(id: str, **kwargs) -> str:
    """Update a {model_name}"""
    instance = await service.update(id, kwargs)
    if instance:
        return instance.model_dump_json(indent=2)
    return f"{{{{ \\"error\\": \\"{model_name} not found\\" }}}}"

@mcp.tool()
async def analyze_{model_name.lower()}(id: str, question: str) -> str:
    """Analyze a {model_name} with LLM"""
    result = await service.analyze(id, question)
    return result

def run_server(port: int = {mcp_def.get('port', 8888)}):
    """Run the MCP server"""
    mcp.run(port=port)

if __name__ == "__main__":
    run_server()
"""
            with open(server_path, "w") as f:
                f.write(server_content)
            
            created_files["mcp"].append(server_path)
        
        elif component_type in ["tool", "resource"]:
            # Tool or Resource definitions are included in the server file
            pass
    
    # Create auth module (simple placeholder)
    auth_file = os.path.join(output_dir, "backend", "auth", "__init__.py")
    os.makedirs(os.path.dirname(auth_file), exist_ok=True)
    
    auth_content = """from typing import List, Optional
from fastapi import Depends, HTTPException, status
from pydantic import BaseModel

class User(BaseModel):
    id: str
    username: str
    roles: List[str] = ["viewer"]

def get_current_user() -> User:
    """Placeholder for getting the current user"""
    # In a real app, this would extract the user from a JWT token or session
    return User(id="user-id", username="user", roles=["admin"])

def requires_role(user: User, required_roles: List[str]) -> None:
    """Check if user has any of the required roles"""
    if not any(role in required_roles for role in user.roles):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to perform this action"
        )
"""
    
    with open(auth_file, "w") as f:
        f.write(auth_content)
    
    # Create cache module (simple placeholder)
    cache_file = os.path.join(output_dir, "backend", "cache", "__init__.py")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    
    cache_content = """from typing import Any, Dict, Optional
from datetime import datetime, timedelta

class Cache:
    """Simple in-memory cache with TTL"""
    
    def __init__(self, ttl: timedelta = timedelta(minutes=5)):
        """Initialize the cache with a default TTL of 5 minutes"""
        self.ttl = ttl
        self._data: Dict[str, Any] = {}
        self._expires: Dict[str, datetime] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get an item from the cache, returning None if expired or not found"""
        if key not in self._data:
            return None
        
        # Check if expired
        if datetime.now() > self._expires[key]:
            self.invalidate(key)
            return None
        
        return self._data[key]
    
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """Set an item in the cache with optional custom TTL"""
        self._data[key] = value
        self._expires[key] = datetime.now() + (ttl or self.ttl)
    
    def invalidate(self, key: str) -> None:
        """Remove an item from the cache"""
        if key in self._data:
            del self._data[key]
        if key in self._expires:
            del self._expires[key]
    
    def invalidate_pattern(self, pattern: str) -> None:
        """Remove all items that match a pattern"""
        for key in list(self._data.keys()):
            if pattern in key:
                self.invalidate(key)
    
    def clear(self) -> None:
        """Clear the entire cache"""
        self._data.clear()
        self._expires.clear()
"""
    
    with open(cache_file, "w") as f:
        f.write(cache_content)
    
    return created_files

def generate_advanced_resource_scaffold(name: str, output_dir: str = None) -> List[str]:
    """Generate an advanced scaffold for a resource
    
    Args:
        name: The name of the resource
        output_dir: Directory to output the scaffold
        
    Returns:
        List of created files
    """
    print(f"Generating advanced resource scaffold for {name}...")
    if output_dir is None:
        output_dir = os.getcwd()
    
    # Create a temporary config file
    config_path = os.path.join(output_dir, f"{name.lower()}_advanced_scaffold_config.py")
    generate_advanced_scaffold_config(name, config_path)
    
    # Create the scaffold from the config
    results = create_advanced_scaffold_from_config(config_path, output_dir)
    
    # Flatten the list of files
    all_files = []
    for category, files in results.items():
        all_files.extend(files)
    
    # Cleanup the temporary config file
    os.unlink(config_path)
    
    return all_files