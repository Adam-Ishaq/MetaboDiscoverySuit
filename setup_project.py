#!/usr/bin/env python3
"""
EM vs Knn - Project Setup Script
Creates the complete directory structure and necessary files for the project.
"""

import os
import sys
from pathlib import Path


def create_directory_structure():
    """Create all necessary directories for the EM vs Knn project."""
    
    # Define directory structure
    directories = [
        # Data directories
        "data/raw",
        "data/processed",
        "data/metadata",
        "data/databases",
        
        # Source code directories
        "src/data_processing",
        "src/preprocessing",
        "src/statistical_analysis",
        "src/ml_models",
        "src/visualization",
        "src/metabolite_identification",
        "src/pathway_analysis",
        "src/utils",
        
        # Notebook directories
        "notebooks",
        
        # Test directories
        "tests",
        
        # Web application directories
        "web_app/pages",
        "web_app/static",
        "web_app/static/css",
        "web_app/static/js",
        
        # Documentation directories
        "docs",
        
        # Results directories
        "results/figures",
        "results/tables",
        "results/reports",
        
        # Other directories
        "logs",
        "cache",
        "models",
    ]
    
    print("=" * 70)
    print("EM vs Knn Project Setup")
    print("=" * 70)
    print()
    
    created_count = 0
    existed_count = 0
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created: {directory}/")
            created_count += 1
        else:
            print(f"○ Exists:  {directory}/")
            existed_count += 1
    
    print()
    print(f"Summary: {created_count} directories created, {existed_count} already existed")
    print()
    
    return directories


def create_gitkeep_files(directories):
    """Create .gitkeep files in empty directories to track them in git."""
    
    print("=" * 70)
    print("Creating .gitkeep files for empty directories")
    print("=" * 70)
    print()
    
    # Directories that should have .gitkeep
    gitkeep_dirs = [
         # Data directories
        "data/raw",
        "data/processed",
        "data/metadata",
        "data/databases",
        
        # Source code directories
        "src/data_processing",
        "src/preprocessing",
        "src/statistical_analysis",
        "src/ml_models",
        "src/visualization",
        "src/metabolite_identification",
        "src/pathway_analysis",
        "src/utils",
        
         # Results directories
        "results/figures",
        "results/tables",
        "results/reports",
       
        "logs",
        "cache",
        "models",
    ]
    
    created_count = 0
    
    for directory in gitkeep_dirs:
        gitkeep_path = Path(directory) / ".gitkeep"
        if not gitkeep_path.exists():
            gitkeep_path.touch()
            print(f"✓ Created: {gitkeep_path}")
            created_count += 1
    
    print()
    print(f"Summary: {created_count} .gitkeep files created")
    print()


def create_init_files():
    """Create __init__.py files in all Python package directories."""
    
    print("=" * 70)
    print("Creating __init__.py files for Python packages")
    print("=" * 70)
    print()
    
    # Directories that need __init__.py
    init_dirs = [
        "src",
        "src/data_processing",
        "src/preprocessing",
        "src/statistical_analysis",
        "src/ml_models",
        "src/visualization",
        "src/metabolite_identification",
        "src/pathway_analysis",
        "src/utils",
        "tests",
    ]
    
    created_count = 0
    
    for directory in init_dirs:
        init_path = Path(directory) / "__init__.py"
        if not init_path.exists():
            with open(init_path, 'w') as f:
                # Write a basic docstring
                module_name = directory.replace('/', '.').replace('src.', '')
                f.write(f'"""\n{module_name} module for MetaboAI\n"""\n')
            print(f"✓ Created: {init_path}")
            created_count += 1
    
    print()
    print(f"Summary: {created_count} __init__.py files created")
    print()


def create_readme_files():
    """Create README files in important directories."""
    
    print("=" * 70)
    print("Creating README files for data directories")
    print("=" * 70)
    print()
    
    readme_content = {
        "data/raw/README.md": """# Raw Data Directory

Place your raw mzML files here.

## Important Notes:
- Raw data files (.mzML) are NOT tracked by git (too large)
- Organize files by experiment or sample group
- Keep a metadata file describing your samples

## Example Structure:
```
raw/
├── experiment_1/
│   ├── sample_001.mzML
│   ├── sample_002.mzML
│   └── ...
└── experiment_2/
    └── ...
```
""",
        
        "data/processed/README.md": """# Processed Data Directory

This directory contains processed feature tables and aligned data.

## Typical Files:
- `feature_table.csv` - Main feature matrix
- `feature_metadata.csv` - Information about detected features
- `sample_qc_metrics.csv` - Quality control metrics

## File Formats:
- CSV files with samples as rows, features as columns
- Or transposed format depending on your preference
""",
        
        "data/metadata/README.md": """# Metadata Directory

Store sample metadata and experimental design information here.

## Required Information:
- Sample names/IDs
- Group assignments (e.g., control vs treatment)
- Batch information
- Any other relevant clinical/experimental variables

## Example Format (CSV):
```
sample_id,group,batch,age,gender
S001,COVID_positive,1,45,M
S002,COVID_negative,1,38,F
...
```
""",
        
        "data/databases/README.md": """# Database Directory

Store metabolite databases and reference files here.

## Suggested Databases:
1. **HMDB** - Human Metabolome Database
2. **METLIN** - Metabolite and Chemical Entity Database
3. **KEGG** - Pathway database
4. **LipidMaps** - Lipid database

## Download Instructions:
Visit respective websites and download database files.
Place them in this directory.
""",
        
        "results/README.md": """# Results Directory

All analysis results, figures, and reports are saved here.

## Subdirectories:
- `figures/` - All generated plots and visualizations
- `tables/` - Statistical results, biomarker rankings
- `reports/` - HTML/PDF reports

## Note:
These files are gitignored. Keep important results backed up separately.
""",
        
        "notebooks/README.md": """# Notebooks Directory

Jupyter notebooks for exploratory analysis and documentation.

## Suggested Notebooks:
1. `01_data_exploration.ipynb` - Initial data inspection
2. `02_preprocessing_tests.ipynb` - Test different preprocessing methods
3. `03_statistical_analysis.ipynb` - Statistical tests and visualizations
4. `04_ml_experiments.ipynb` - Machine learning model testing
5. `05_results_visualization.ipynb` - Final result visualizations

## Best Practices:
- Clear outputs before committing
- Document your analysis steps
- Include markdown explanations
""",
        
        "tests/README.md": """# Tests Directory

Unit tests and integration tests for the codebase.

## Running Tests:
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_data_processing.py

# Run with coverage
pytest --cov=src tests/
```

## Test Structure:
- `test_*.py` - Test files
- Mirror the src/ directory structure
""",
    }
    
    created_count = 0
    
    for filepath, content in readme_content.items():
        readme_path = Path(filepath)
        if not readme_path.exists():
            readme_path.parent.mkdir(parents=True, exist_ok=True)
            with open(readme_path, 'w') as f:
                f.write(content)
            print(f"✓ Created: {filepath}")
            created_count += 1
    
    print()
    print(f"Summary: {created_count} README files created")
    print()


def create_example_metadata():
    """Create an example metadata template file."""
    
    print("=" * 70)
    print("Creating example metadata template")
    print("=" * 70)
    print()
    
    metadata_template = """sample_id,group,batch,filename
S001,group_A,1,sample_001.mzML
S002,group_A,1,sample_002.mzML
S003,group_B,1,sample_003.mzML
S004,group_B,2,sample_004.mzML
"""
    
    template_path = Path("data/metadata/metadata_template.csv")
    if not template_path.exists():
        with open(template_path, 'w') as f:
            f.write(metadata_template)
        print(f"✓ Created: {template_path}")
        print()


def print_next_steps():
    """Print next steps for the user."""
    
    print("=" * 70)
    print("Setup Complete! 🎉")
    print("=" * 70)
    print()
    print("Next Steps:")
    print()
    print("1. Review the created directory structure")
    print("2. Place your mzML files in data/raw/")
    print("3. Create your metadata file in data/metadata/")
    print("4. Copy config.yaml to your project root")
    print("5. Install dependencies: pip install -r requirements.txt")
    print("6. Start coding the mzML parser!")
    print()
    print("To commit these changes to git:")
    print("  git add .")
    print("  git commit -m 'Set up project directory structure'")
    print("  git push origin main")
    print()
    print("=" * 70)


def main():
    """Main setup function."""
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists() and not Path("README.md").exists():
        print("Warning: This doesn't appear to be the project root directory.")
        print("Please run this script from the MetaboAI project root.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            sys.exit(0)
    
    try:
        # Create directory structure
        directories = create_directory_structure()
        
        # Create .gitkeep files
        create_gitkeep_files(directories)
        
        # Create __init__.py files
        create_init_files()
        
        # Create README files
        create_readme_files()
        
        # Create example metadata
        create_example_metadata()
        
        # Print next steps
        print_next_steps()
        
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
