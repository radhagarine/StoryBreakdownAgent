import { useState, useEffect } from 'react';
import "./App.css";

// Get backend URL from environment variable
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

// Test component to verify backend connection - this will be replaced by your UI
function App() {
  const [apiStatus, setApiStatus] = useState('Checking...');
  const [error, setError] = useState(null);

  useEffect(() => {
    // Test the API connection
    fetch(`${BACKEND_URL}/api`)
      .then(response => {
        if (!response.ok) {
          throw new Error(`API request failed with status ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        setApiStatus(`Connected: ${data.message}`);
        setError(null);
      })
      .catch(err => {
        setApiStatus('Connection failed');
        setError(err.message);
      });
  }, []);

  return (
    <div className="flex min-h-screen bg-gray-100">
      <div className="w-full max-w-4xl mx-auto p-8">
        <header className="text-center mb-10">
          <h1 className="text-3xl font-bold text-gray-800 mb-4">
            Movie Script Breakdown & Image Prompt Generator
          </h1>
          <p className="text-gray-600 mb-2">
            Backend API Status: 
            <span className={`ml-2 font-semibold ${apiStatus.includes('Connected') ? 'text-green-600' : 'text-red-600'}`}>
              {apiStatus}
            </span>
          </p>
          {error && <p className="text-red-500">{error}</p>}
          <p className="text-gray-500 text-sm mt-4">
            Note: This is a test interface. The actual UI will be provided separately.
          </p>
        </header>
        
        <div className="bg-white shadow-xl rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">API Endpoints Available:</h2>
          
          <div className="space-y-2 mb-6">
            <EndpointItem 
              method="POST" 
              path="/api/scripts/upload" 
              description="Upload and process a script file" 
            />
            <EndpointItem 
              method="GET" 
              path="/api/scripts" 
              description="Get list of all scripts" 
            />
            <EndpointItem 
              method="GET" 
              path="/api/scripts/{script_id}" 
              description="Get script details" 
            />
            <EndpointItem 
              method="GET" 
              path="/api/scripts/{script_id}/characters" 
              description="Get all characters from a script" 
            />
            <EndpointItem 
              method="GET" 
              path="/api/scripts/{script_id}/scenes" 
              description="Get all scenes from a script" 
            />
            <EndpointItem 
              method="GET" 
              path="/api/scripts/{script_id}/shots" 
              description="Get all shots from a script" 
            />
            <EndpointItem 
              method="GET" 
              path="/api/characters/{character_id}" 
              description="Get details of a specific character" 
            />
            <EndpointItem 
              method="PUT" 
              path="/api/characters/{character_id}" 
              description="Update character details" 
            />
            <EndpointItem 
              method="POST" 
              path="/api/characters/{character_id}/generate-prompt" 
              description="Generate image prompt for a character" 
            />
            <EndpointItem 
              method="GET" 
              path="/api/scenes/{scene_id}" 
              description="Get details of a specific scene" 
            />
            <EndpointItem 
              method="PUT" 
              path="/api/scenes/{scene_id}" 
              description="Update scene details" 
            />
            <EndpointItem 
              method="POST" 
              path="/api/scenes/{scene_id}/generate-prompt" 
              description="Generate image prompt for a scene" 
            />
            <EndpointItem 
              method="GET" 
              path="/api/shots/{shot_id}" 
              description="Get details of a specific shot" 
            />
            <EndpointItem 
              method="PUT" 
              path="/api/shots/{shot_id}" 
              description="Update shot details" 
            />
            <EndpointItem 
              method="POST" 
              path="/api/shots/{shot_id}/generate-prompt" 
              description="Generate image prompt for a shot" 
            />
            <EndpointItem 
              method="GET" 
              path="/api/prompts/{entity_type}/{entity_id}" 
              description="Get all prompts for a specific entity" 
            />
          </div>
          
          <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
            <h3 className="text-md font-semibold text-blue-800 mb-2">Integration Notes:</h3>
            <ul className="list-disc pl-5 space-y-1 text-blue-700">
              <li>Your UI should connect to these endpoints using the provided URL.</li>
              <li>For file upload, use a multipart/form-data request with 'file' and optional 'title' fields.</li>
              <li>All responses are in JSON format and include appropriate HTTP status codes.</li>
              <li>Errors are returned with detailed messages to aid debugging.</li>
              <li>Image prompts are automatically generated during script parsing.</li>
              <li>Prompts can be regenerated using the dedicated endpoints.</li>
            </ul>
          </div>
        </div>
        
        <footer className="mt-8 text-center text-gray-500 text-sm">
          <p>Movie Script Breakdown API - 2025</p>
        </footer>
      </div>
    </div>
  );
}

// Helper component for displaying endpoint information
function EndpointItem({ method, path, description }) {
  return (
    <div className="flex items-start border-b border-gray-100 pb-2">
      <span className={`
        inline-block px-2 py-1 text-xs font-bold rounded mr-3 min-w-[60px] text-center
        ${method === 'GET' ? 'bg-green-100 text-green-800' : 
          method === 'POST' ? 'bg-blue-100 text-blue-800' : 
          method === 'PUT' ? 'bg-yellow-100 text-yellow-800' : 
          method === 'DELETE' ? 'bg-red-100 text-red-800' : 'bg-gray-100 text-gray-800'}
      `}>
        {method}
      </span>
      <div>
        <code className="text-sm font-mono bg-gray-100 px-2 py-1 rounded">{path}</code>
        <p className="text-gray-600 text-sm mt-1">{description}</p>
      </div>
    </div>
  );
}

export default App;
