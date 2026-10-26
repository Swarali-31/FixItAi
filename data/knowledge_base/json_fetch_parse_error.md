# Troubleshooting JSON File Fetch and Parse Errors in Web Applications

## Problem Summary
When developing web applications, attempting to fetch, read, or parse data from a `.json` file frequently fails with errors such as `404 Not Found`, `CORS Policy Blocked`, or `SyntaxError: Unexpected token '<', "<!DOCTYPE "... is not valid JSON`.

## Root Cause
1. **Frontend Relative File Path Misplacement**: In Single Page Applications (React, Vite, Next.js), calling `fetch('./data.json')` attempts to load the file relative to the current route URL rather than the web server's static root. Files must be located in the `public/` folder and fetched as `fetch('/data.json')`.
2. **CORS Restrictions on Local File Access**: Opening HTML files via `file:///` causes browsers to block `fetch()` or `XMLHttpRequest` under the Same-Origin Policy.
3. **Invalid JSON Formatting**: Trailing commas, single quotes around keys/values, or unquoted keys fail `JSON.parse()`.
4. **Backend File System Paths**: In Node.js or Python, relative paths are resolved relative to the current working directory (`process.cwd()` or `os.getcwd()`), not the script file location.

## Action Steps
1. In React / Vite, move the JSON file into the root `public/` directory and fetch using an absolute path: `fetch('/data.json')`.
2. Always check `if (!response.ok)` before invoking `await response.json()`.
3. If using Node.js, use `path.join(__dirname, 'data.json')` with `fs.promises.readFile()`.
4. If using Python, open with `utf-8` encoding: `with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)`.
5. Validate JSON syntax using a linter or `JSON.parse()` to ensure double quotes and no trailing commas.

## Verified Code Fix

```javascript
// Frontend (React / Vite / Vanilla JS):
async function loadJsonData() {
  try {
    // 1. Fetch from root public/ folder (e.g. public/data.json)
    const response = await fetch('/data.json', {
      headers: { 'Accept': 'application/json' }
    });
    
    // 2. Guard against 404 / HTML error pages
    if (!response.ok) {
      throw new Error(`Failed to load JSON: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    console.log('JSON data loaded successfully:', data);
    return data;
  } catch (error) {
    console.error('JSON fetch error:', error);
  }
}
```

```python
# Python Alternative (with UTF-8):
import json
from pathlib import Path

def load_local_json(filename="data.json"):
    path = Path(__file__).parent / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
```
