"""
Test script for mzML parser - Use this to test with your COVID data
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing.mzml_parser import MZMLParser


def test_single_file(mzml_path: str):
    """
    Test parser with a single mzML file.
    
    Args:
        mzml_path (str): Path to mzML file
    """
    print("=" * 70)
    print("Testing mzML Parser with Single File")
    print("=" * 70)
    print()
    
    # Initialize parser
    print("Step 1: Initializing parser...")
    parser = MZMLParser()
    print("✓ Parser initialized")
    print()
    
    # Load file
    print(f"Step 2: Loading file: {mzml_path}")
    success = parser.load_file(mzml_path)
    
    if not success:
        print("✗ Failed to load file")
        return
    
    print("✓ File loaded successfully")
    print()
    
    # Print file summary
    print("Step 3: File Summary")
    parser.print_file_summary()
    print()
    
    # Get file info programmatically
    file_info = parser.get_file_info()
    
    # Extract a single spectrum for inspection
    print("Step 4: Extracting first MS1 spectrum...")
    ms1_spectra = parser.get_ms1_spectra()
    
    if len(ms1_spectra) > 0:
        first_spectrum = ms1_spectra[0]
        print(f"✓ First spectrum extracted")
        print(f"  - Scan index: {first_spectrum['scan_index']}")
        print(f"  - Retention time: {first_spectrum['retention_time']:.2f} seconds")
        print(f"  - Number of peaks: {first_spectrum['n_peaks']}")
        print(f"  - Base peak m/z: {first_spectrum['base_peak_mz']:.4f}")
        print(f"  - TIC: {first_spectrum['tic']:.2e}")
    else:
        print("✗ No MS1 spectra found")
        return
    
    print()
    
    # Extract features
    print("Step 5: Extracting all features...")
    features_df = parser.extract_features()
    
    if len(features_df) > 0:
        print(f"✓ Extracted {len(features_df)} features")
        print()
        print("Feature statistics:")
        print(f"  - m/z range: {features_df['mz'].min():.2f} - {features_df['mz'].max():.2f}")
        print(f"  - RT range: {features_df['rt'].min():.2f} - {features_df['rt'].max():.2f} seconds")
        print(f"  - Intensity range: {features_df['intensity'].min():.2e} - {features_df['intensity'].max():.2e}")
        print(f"  - Number of scans: {features_df['scan_index'].nunique()}")
        print()
        
        # Show first few features
        print("First 10 features:")
        print(features_df.head(10))
    else:
        print("✗ No features extracted")
        return
    
    print()
    
    # Extract chromatograms
    print("Step 6: Extracting chromatograms...")
    rt_tic, tic = parser.get_tic_chromatogram()
    rt_bpc, bpc = parser.get_bpc_chromatogram()
    
    print(f"✓ TIC extracted: {len(tic)} points")
    print(f"✓ BPC extracted: {len(bpc)} points")
    print()
    
    # Plot results
    print("Step 7: Creating visualizations...")
    plot_results(parser, features_df, rt_tic, tic, rt_bpc, bpc, first_spectrum)
    print("✓ Plots created")
    print()
    
    # Save features
    output_path = Path("results") / "test_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)
    print(f"Step 8: Features saved to: {output_path}")
    print()
    
    print("=" * 70)
    print("✓ Test completed successfully!")
    print("=" * 70)


def plot_results(parser, features_df, rt_tic, tic, rt_bpc, bpc, first_spectrum):
    """Create visualization plots for the parsed data."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"mzML Parsing Results: {parser.file_info['filename']}", 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: TIC Chromatogram
    ax1 = axes[0, 0]
    ax1.plot(rt_tic / 60, tic, color='blue', linewidth=1)
    ax1.set_xlabel('Retention Time (minutes)')
    ax1.set_ylabel('Total Ion Current')
    ax1.set_title('Total Ion Chromatogram (TIC)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: BPC Chromatogram
    ax2 = axes[0, 1]
    ax2.plot(rt_bpc / 60, bpc, color='red', linewidth=1)
    ax2.set_xlabel('Retention Time (minutes)')
    ax2.set_ylabel('Base Peak Intensity')
    ax2.set_title('Base Peak Chromatogram (BPC)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: First spectrum (mass spectrum)
    ax3 = axes[1, 0]
    mz = first_spectrum['mz']
    intensity = first_spectrum['intensity']
    ax3.vlines(mz, 0, intensity, color='black', linewidth=0.5)
    ax3.set_xlabel('m/z')
    ax3.set_ylabel('Intensity')
    ax3.set_title(f'First MS1 Spectrum (RT: {first_spectrum["retention_time"]/60:.2f} min)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Feature density heatmap (2D: RT vs m/z)
    ax4 = axes[1, 1]
    
    # Create 2D histogram
    rt_bins = 50
    mz_bins = 50
    
    H, xedges, yedges = np.histogram2d(
        features_df['rt'] / 60,
        features_df['mz'],
        bins=[rt_bins, mz_bins],
        weights=features_df['intensity']
    )
    
    im = ax4.imshow(H.T, origin='lower', aspect='auto', 
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    cmap='viridis', interpolation='nearest')
    ax4.set_xlabel('Retention Time (minutes)')
    ax4.set_ylabel('m/z')
    ax4.set_title('Feature Density Map')
    plt.colorbar(im, ax=ax4, label='Intensity')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path("results/figures")
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / "mzml_parsing_results.png", dpi=300, bbox_inches='tight')
    print(f"  - Figure saved to: {output_path / 'mzml_parsing_results.png'}")
    
    # Show plot
    plt.show()


def test_multiple_files(mzml_dir: str):
    """
    Test parser with multiple mzML files in a directory.
    
    Args:
        mzml_dir (str): Directory containing mzML files
    """
    print("=" * 70)
    print("Testing mzML Parser with Multiple Files")
    print("=" * 70)
    print()
    
    # Find all mzML files
    mzml_files = list(Path(mzml_dir).glob("*.mzML"))
    
    if len(mzml_files) == 0:
        print(f"No mzML files found in: {mzml_dir}")
        return
    
    print(f"Found {len(mzml_files)} mzML files")
    print()
    
    # Parse each file
    from src.data_processing.mzml_parser import parse_multiple_files
    
    results = parse_multiple_files(
        [str(f) for f in mzml_files],
        output_dir="results/individual_features"
    )
    
    print()
    print("Summary:")
    print("-" * 70)
    
    for filename, features in results.items():
        print(f"{filename}:")
        print(f"  - Features: {len(features)}")
        if len(features) > 0:
            print(f"  - m/z range: {features['mz'].min():.2f} - {features['mz'].max():.2f}")
            print(f"  - RT range: {features['rt'].min():.2f} - {features['rt'].max():.2f}")
    
    print("=" * 70)


def main():
    """Main test function."""
    
    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║              mzML Parser - Test Script                            ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print()
        print("  Test single file:")
        print("    python test_mzml_parser.py path/to/file.mzML")
        print()
        print("  Test multiple files:")
        print("    python test_mzml_parser.py path/to/directory/")
        print()
        print("Example:")
        print("  python test_mzml_parser.py data/raw/covid_sample_001.mzML")
        print()
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # Check if it's a file or directory
    path = Path(input_path)
    
    if path.is_file():
        # Test single file
        test_single_file(str(path))
    elif path.is_dir():
        # Test multiple files
        test_multiple_files(str(path))
    else:
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
