# Data Processing Module

This module handles the parsing and processing of raw LC-MS data files (mzML format).

## Files

### `mzml_parser.py`
Main parser for reading mzML files and extracting features.

**Key Classes:**
- `MZMLParser`: Main parser class for mzML files

**Key Functions:**
- `load_file()`: Load an mzML file
- `extract_spectrum()`: Extract a single spectrum
- `get_ms1_spectra()`: Get all MS1 spectra
- `extract_features()`: Extract all features as DataFrame
- `get_tic_chromatogram()`: Get Total Ion Chromatogram
- `get_bpc_chromatogram()`: Get Base Peak Chromatogram
- `export_features_to_csv()`: Export features to CSV

## Usage

### Basic Usage

```python
from src.data_processing.mzml_parser import MZMLParser

# Initialize parser
parser = MZMLParser()

# Load mzML file
parser.load_file('data/raw/sample.mzML')

# Print file summary
parser.print_file_summary()

# Extract features
features = parser.extract_features()

# Export to CSV
parser.export_features_to_csv('output_features.csv')
```

### Advanced Configuration

```python
# Custom configuration
config = {
    'snr_threshold': 5.0,
    'intensity_threshold': 5000.0,
    'mass_error_ppm': 5.0
}

parser = MZMLParser(config=config)
```

### Processing Multiple Files

```python
from src.data_processing.mzml_parser import parse_multiple_files

mzml_files = [
    'data/raw/sample_001.mzML',
    'data/raw/sample_002.mzML',
    'data/raw/sample_003.mzML'
]

results = parse_multiple_files(
    mzml_files,
    output_dir='results/features'
)
```

## Testing

Use the test script to verify the parser works with your data:

```bash
# Test single file
python test_mzml_parser.py data/raw/covid_sample.mzML

# Test multiple files
python test_mzml_parser.py data/raw/
```

## Output Format

### Features DataFrame
The extracted features are returned as a pandas DataFrame with columns:

| Column | Description |
|--------|-------------|
| `mz` | Mass-to-charge ratio |
| `rt` | Retention time (seconds) |
| `intensity` | Peak intensity |
| `scan_index` | Index of the scan where peak was detected |

### Example Output
```
       mz          rt     intensity  scan_index
0   100.0523    45.23    12345.67          12
1   150.2341    45.23    23456.78          12
2   200.8765    45.23     8901.23          12
3   100.0524    46.15    11234.56          13
...
```

## File Information

The parser extracts the following metadata:

- **filename**: Name of the mzML file
- **n_spectra**: Total number of spectra
- **ms_levels**: MS levels present (e.g., [1, 2])
- **polarity**: Ionization mode (positive/negative)
- **spectrum_type**: Data type (centroid/profile)
- **rt_range**: Retention time range
- **mz_min/mz_max**: m/z range

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `snr_threshold` | 3.0 | Signal-to-noise ratio threshold |
| `intensity_threshold` | 1000.0 | Minimum peak intensity |
| `mass_error_ppm` | 10.0 | Mass accuracy in ppm |
| `rt_tolerance` | 30.0 | RT tolerance for alignment (seconds) |
| `mz_tolerance` | 0.01 | m/z tolerance for alignment (Da) |

## Next Steps

After parsing individual files, the next steps are:

1. **Peak Alignment** (`alignment.py`) - Align features across samples
2. **Feature Extraction** (`feature_extraction.py`) - Create unified feature table
3. **Quality Control** - Filter and validate features

## Requirements

- Python 3.9+
- PyOpenMS (conda install -c bioconda pyopenms)
- pandas
- numpy
- matplotlib (for visualization)
- tqdm (for progress bars)

## Troubleshooting

### PyOpenMS Import Error
```bash
# Install PyOpenMS via conda (recommended)
conda install -c bioconda pyopenms
```

### File Not Loading
- Check file path is correct
- Verify file is valid mzML format
- Ensure file has .mzML or .mzXML extension

### No Features Extracted
- Lower `intensity_threshold` in config
- Check if file contains MS1 data
- Verify file is centroided (not profile)

### Memory Issues
- Process files one at a time
- Increase intensity threshold to reduce features
- Use batch processing for large datasets

## Contact

For issues or questions about the parser, please open a GitHub issue.
