"""
Scan all Python files and extract third-party imports.
"""
import ast
import sys
from pathlib import Path
from collections import defaultdict

def get_imports_from_file(filepath):
    """Extract all imports from a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(filepath))
        
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
        
        return imports
    except Exception as e:
        print(f"Error parsing {filepath}: {e}", file=sys.stderr)
        return set()

def main():
    src_dir = Path(__file__).parent / 'src'
    all_imports = set()
    
    # Scan all Python files
    for py_file in src_dir.rglob('*.py'):
        imports = get_imports_from_file(py_file)
        all_imports.update(imports)
    
    # Filter out standard library and local imports
    stdlib = sys.stdlib_module_names
    third_party = sorted([
        imp for imp in all_imports 
        if imp and imp not in stdlib and not imp.startswith('pb_studio')
    ])
    
    print("=== THIRD-PARTY IMPORTS GEFUNDEN ===\n")
    for imp in third_party:
        print(imp)
    
    print(f"\n=== GESAMT: {len(third_party)} Pakete ===")

if __name__ == '__main__':
    main()
