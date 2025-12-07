"""
MetaboAI - mzML Parser Module
Parses LC-MS data files in mzML format and extracts features for analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm

try:
    import pyopenms as oms
    PYOPENMS_AVAILABLE = True
except ImportError:
    PYOPENMS_AVAILABLE = False
    print("Warning: PyOpenMS not available. Install with: conda install -c bioconda pyopenms")


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MZMLParser:
    """
    Parser for mzML files from LC-MS experiments.
    
    This class handles:
    - Reading mzML files
    - Extracting spectra and metadata
    - Peak detection
    - Feature extraction
    
    Attributes:
        config (dict): Configuration parameters for parsing
        experiment (MSExperiment): Loaded mass spectrometry experiment
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the mzML parser.
        
        Args:
            config (dict, optional): Configuration dictionary with parameters:
                - snr_threshold: Signal-to-noise ratio threshold (default: 3.0)
                - intensity_threshold: Minimum peak intensity (default: 1000.0)
                - mass_error_ppm: Mass accuracy in ppm (default: 10.0)
        """
        if not PYOPENMS_AVAILABLE:
            raise ImportError(
                "PyOpenMS is required for mzML parsing. "
                "Install with: conda install -c bioconda pyopenms"
            )
        
        # Default configuration
        self.config = {
            'snr_threshold': 3.0,
            'intensity_threshold': 1000.0,
            'mass_error_ppm': 10.0,
            'rt_tolerance': 30.0,
            'mz_tolerance': 0.01
        }
        
        # Update with user config
        if config:
            self.config.update(config)
        
        self.experiment = None
        self.file_info = {}
        
        logger.info("MZMLParser initialized with config: %s", self.config)
    
    def load_file(self, mzml_path: str) -> bool:
        """
        Load an mzML file into memory.
        
        Args:
            mzml_path (str): Path to the mzML file
            
        Returns:
            bool: True if successful, False otherwise
        """
        mzml_path = Path(mzml_path)
        
        if not mzml_path.exists():
            logger.error(f"File not found: {mzml_path}")
            return False
        
        if not mzml_path.suffix.lower() in ['.mzml', '.mzxml']:
            logger.error(f"Invalid file format: {mzml_path.suffix}")
            return False
        
        try:
            logger.info(f"Loading mzML file: {mzml_path}")
            
            # Create MSExperiment object
            self.experiment = oms.MSExperiment()
            
            # Load the file
            oms.MzMLFile().load(str(mzml_path), self.experiment)
            
            # Extract file information
            self._extract_file_info(mzml_path)
            
            logger.info(f"Successfully loaded {self.file_info['n_spectra']} spectra")
            return True
            
        except Exception as e:
            logger.error(f"Error loading mzML file: {e}")
            return False
    
    def _extract_file_info(self, file_path: Path):
        """
        Extract metadata and summary information from the loaded experiment.
        
        Args:
            file_path (Path): Path to the mzML file
        """
        self.file_info = {
            'filename': file_path.name,
            'filepath': str(file_path),
            'n_spectra': self.experiment.getNrSpectra(),
            'n_chromatograms': self.experiment.getNrChromatograms(),
        }
        
        if self.experiment.getNrSpectra() > 0:
            # Get RT range
            rts = [spec.getRT() for spec in self.experiment]
            self.file_info['rt_min'] = min(rts)
            self.file_info['rt_max'] = max(rts)
            self.file_info['rt_range'] = max(rts) - min(rts)
            
            # Get m/z range (from first spectrum)
            first_spec = self.experiment[0]
            if first_spec.size() > 0:
                mz, _ = first_spec.get_peaks()
                self.file_info['mz_min'] = mz[0] if len(mz) > 0 else 0
                self.file_info['mz_max'] = mz[-1] if len(mz) > 0 else 0
            
            # Get polarity
            polarity = first_spec.getInstrumentSettings().getPolarity()
            if polarity == oms.IonSource.Polarity.POSITIVE:
                self.file_info['polarity'] = 'positive'
            elif polarity == oms.IonSource.Polarity.NEGATIVE:
                self.file_info['polarity'] = 'negative'
            else:
                self.file_info['polarity'] = 'unknown'
            
            # Get MS levels present
            ms_levels = set(spec.getMSLevel() for spec in self.experiment)
            self.file_info['ms_levels'] = sorted(list(ms_levels))
            
            # Check if data is centroided or profile
            spec_type = first_spec.getType()
            if spec_type == oms.SpectrumSettings.SpectrumType.CENTROID:
                self.file_info['spectrum_type'] = 'centroid'
            elif spec_type == oms.SpectrumSettings.SpectrumType.PROFILE:
                self.file_info['spectrum_type'] = 'profile'
            else:
                self.file_info['spectrum_type'] = 'unknown'
        
        logger.info(f"File info extracted: {self.file_info}")
    
    def get_file_info(self) -> Dict:
        """
        Get information about the loaded mzML file.
        
        Returns:
            dict: Dictionary containing file metadata
        """
        return self.file_info.copy()
    
    def print_file_summary(self):
        """Print a human-readable summary of the loaded file."""
        if not self.file_info:
            print("No file loaded.")
            return
        
        print("=" * 70)
        print("mzML File Summary")
        print("=" * 70)
        print(f"Filename:        {self.file_info.get('filename', 'N/A')}")
        print(f"Total Spectra:   {self.file_info.get('n_spectra', 0)}")
        print(f"MS Levels:       {self.file_info.get('ms_levels', [])}")
        print(f"Polarity:        {self.file_info.get('polarity', 'unknown')}")
        print(f"Spectrum Type:   {self.file_info.get('spectrum_type', 'unknown')}")
        print(f"RT Range:        {self.file_info.get('rt_min', 0):.2f} - {self.file_info.get('rt_max', 0):.2f} seconds")
        print(f"m/z Range:       {self.file_info.get('mz_min', 0):.2f} - {self.file_info.get('mz_max', 0):.2f}")
        print("=" * 70)
    
    def extract_spectrum(self, scan_index: int) -> Optional[Dict]:
        """
        Extract a single spectrum by index.
        
        Args:
            scan_index (int): Index of the spectrum to extract
            
        Returns:
            dict: Dictionary containing spectrum data or None if invalid
        """
        if self.experiment is None:
            logger.error("No file loaded. Call load_file() first.")
            return None
        
        if scan_index < 0 or scan_index >= self.experiment.getNrSpectra():
            logger.error(f"Invalid scan index: {scan_index}")
            return None
        
        spectrum = self.experiment[scan_index]
        
        # Get peaks
        mz, intensity = spectrum.get_peaks()
        
        # Extract metadata
        spec_data = {
            'scan_index': scan_index,
            'retention_time': spectrum.getRT(),
            'ms_level': spectrum.getMSLevel(),
            'mz': mz,
            'intensity': intensity,
            'n_peaks': len(mz),
            'base_peak_mz': 0.0,
            'base_peak_intensity': 0.0,
            'tic': np.sum(intensity)
        }
        
        # Get base peak info
        if len(intensity) > 0:
            max_idx = np.argmax(intensity)
            spec_data['base_peak_mz'] = mz[max_idx]
            spec_data['base_peak_intensity'] = intensity[max_idx]
        
        return spec_data
    
    def get_ms1_spectra(self) -> List[Dict]:
        """
        Extract all MS1 spectra from the experiment.
        
        Returns:
            list: List of dictionaries containing MS1 spectrum data
        """
        if self.experiment is None:
            logger.error("No file loaded. Call load_file() first.")
            return []
        
        ms1_spectra = []
        
        logger.info("Extracting MS1 spectra...")
        for i, spectrum in enumerate(tqdm(self.experiment, desc="Processing spectra")):
            if spectrum.getMSLevel() == 1:
                spec_data = self.extract_spectrum(i)
                if spec_data:
                    ms1_spectra.append(spec_data)
        
        logger.info(f"Extracted {len(ms1_spectra)} MS1 spectra")
        return ms1_spectra
    
    def detect_peaks(self, spectrum_data: Dict, 
                    intensity_threshold: Optional[float] = None) -> Dict:
        """
        Detect peaks in a spectrum based on intensity threshold.
        
        Args:
            spectrum_data (dict): Spectrum data from extract_spectrum()
            intensity_threshold (float, optional): Minimum intensity for peaks
            
        Returns:
            dict: Dictionary with filtered peaks
        """
        if intensity_threshold is None:
            intensity_threshold = self.config['intensity_threshold']
        
        mz = spectrum_data['mz']
        intensity = spectrum_data['intensity']
        
        # Filter by intensity threshold
        mask = intensity >= intensity_threshold
        
        filtered_data = spectrum_data.copy()
        filtered_data['mz'] = mz[mask]
        filtered_data['intensity'] = intensity[mask]
        filtered_data['n_peaks'] = np.sum(mask)
        filtered_data['threshold_used'] = intensity_threshold
        
        return filtered_data
    
    def extract_features(self, ms_level: int = 1) -> pd.DataFrame:
        """
        Extract features from all spectra of specified MS level.
        
        This creates a table of detected features with their properties.
        
        Args:
            ms_level (int): MS level to extract (default: 1)
            
        Returns:
            pd.DataFrame: DataFrame with columns [mz, rt, intensity, scan_idx]
        """
        if self.experiment is None:
            logger.error("No file loaded. Call load_file() first.")
            return pd.DataFrame()
        
        features = []
        
        logger.info(f"Extracting features from MS{ms_level} spectra...")
        
        for i, spectrum in enumerate(tqdm(self.experiment, desc="Extracting features")):
            if spectrum.getMSLevel() == ms_level:
                mz, intensity = spectrum.get_peaks()
                rt = spectrum.getRT()
                
                # Filter by intensity threshold
                mask = intensity >= self.config['intensity_threshold']
                
                # Create feature entries
                for m, intens in zip(mz[mask], intensity[mask]):
                    features.append({
                        'mz': m,
                        'rt': rt,
                        'intensity': intens,
                        'scan_index': i
                    })
        
        df = pd.DataFrame(features)
        
        if len(df) > 0:
            logger.info(f"Extracted {len(df)} features from {len(df['scan_index'].unique())} spectra")
        else:
            logger.warning("No features extracted. Check intensity threshold.")
        
        return df
    
    def get_tic_chromatogram(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract Total Ion Chromatogram (TIC).
        
        Returns:
            tuple: (retention_times, total_ion_currents)
        """
        if self.experiment is None:
            logger.error("No file loaded. Call load_file() first.")
            return np.array([]), np.array([])
        
        rts = []
        tics = []
        
        for spectrum in self.experiment:
            if spectrum.getMSLevel() == 1:
                _, intensity = spectrum.get_peaks()
                rts.append(spectrum.getRT())
                tics.append(np.sum(intensity))
        
        return np.array(rts), np.array(tics)
    
    def get_bpc_chromatogram(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract Base Peak Chromatogram (BPC).
        
        Returns:
            tuple: (retention_times, base_peak_intensities)
        """
        if self.experiment is None:
            logger.error("No file loaded. Call load_file() first.")
            return np.array([]), np.array([])
        
        rts = []
        bpcs = []
        
        for spectrum in self.experiment:
            if spectrum.getMSLevel() == 1:
                _, intensity = spectrum.get_peaks()
                rts.append(spectrum.getRT())
                bpcs.append(np.max(intensity) if len(intensity) > 0 else 0)
        
        return np.array(rts), np.array(bpcs)
    
    def export_features_to_csv(self, output_path: str, ms_level: int = 1):
        """
        Extract features and export to CSV file.
        
        Args:
            output_path (str): Path for output CSV file
            ms_level (int): MS level to extract (default: 1)
        """
        features_df = self.extract_features(ms_level=ms_level)
        
        if len(features_df) > 0:
            features_df.to_csv(output_path, index=False)
            logger.info(f"Features exported to: {output_path}")
        else:
            logger.warning("No features to export.")


def parse_multiple_files(mzml_files: List[str], 
                        config: Optional[Dict] = None,
                        output_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Parse multiple mzML files and extract features from each.
    
    Args:
        mzml_files (list): List of paths to mzML files
        config (dict, optional): Configuration for parser
        output_dir (str, optional): Directory to save individual feature files
        
    Returns:
        dict: Dictionary mapping filenames to feature DataFrames
    """
    parser = MZMLParser(config)
    results = {}
    
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for mzml_file in mzml_files:
        logger.info(f"Processing: {mzml_file}")
        
        if parser.load_file(mzml_file):
            # Extract features
            features = parser.extract_features()
            
            filename = Path(mzml_file).stem
            results[filename] = features
            
            # Optionally save to file
            if output_dir and len(features) > 0:
                output_path = Path(output_dir) / f"{filename}_features.csv"
                features.to_csv(output_path, index=False)
                logger.info(f"Saved features to: {output_path}")
    
    return results


# Example usage and testing
if __name__ == "__main__":
    print("MZMLParser Module")
    print("=" * 70)
    print()
    print("This module provides functionality to parse mzML files from LC-MS.")
    print()
    print("Example usage:")
    print()
    print("  from src.data_processing.mzml_parser import MZMLParser")
    print()
    print("  # Initialize parser")
    print("  parser = MZMLParser()")
    print()
    print("  # Load mzML file")
    print("  parser.load_file('data/raw/sample.mzML')")
    print()
    print("  # Print file summary")
    print("  parser.print_file_summary()")
    print()
    print("  # Extract features")
    print("  features = parser.extract_features()")
    print()
    print("  # Export to CSV")
    print("  parser.export_features_to_csv('output_features.csv')")
    print()
    print("=" * 70)
