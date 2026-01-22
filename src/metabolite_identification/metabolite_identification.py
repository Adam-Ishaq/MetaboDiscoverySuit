"""
MetaboAI - Metabolite Identification Module
Identifies metabolites by matching m/z values to HMDB database.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import xml.etree.ElementTree as ET
import logging
from tqdm import tqdm
import gzip

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HMDBParser:
    """
    Parser for HMDB XML database.
    """
    
    def __init__(self):
        """Initialize HMDB parser."""
        self.metabolites = []
        logger.info("HMDBParser initialized")
    
    def parse_hmdb_xml(self, hmdb_path: str, max_metabolites: Optional[int] = None) -> pd.DataFrame:
        """
        Parse HMDB XML file and extract metabolite information.
        
        Args:
            hmdb_path (str): Path to hmdb.xml or hmdb.xml.gz file
            max_metabolites (int, optional): Limit parsing (for testing)
            
        Returns:
            DataFrame: Metabolite database
        """
        hmdb_path = Path(hmdb_path)
        
        if not hmdb_path.exists():
            logger.error(f"HMDB file not found: {hmdb_path}")
            return pd.DataFrame()
        
        logger.info(f"Parsing HMDB database: {hmdb_path.name}")
        logger.info("This may take several minutes for large databases...")
        
        # Open file (handle both .xml and .xml.gz)
        if hmdb_path.suffix == '.gz':
            import gzip
            file_handle = gzip.open(hmdb_path, 'rt', encoding='utf-8')
        else:
            file_handle = open(hmdb_path, 'r', encoding='utf-8')
        
        metabolites = []
        current_metabolite = {}
        current_tag = None
        in_metabolite = False
        in_taxonomy = False
        
        try:
            logger.info("Starting line-by-line parsing...")
            count = 0
            line_count = 0
            
            for line in tqdm(file_handle, desc="Parsing HMDB", unit=" lines"):
                line_count += 1
                line = line.strip()
                
                # Check for metabolite start
                if '<metabolite>' in line:
                    in_metabolite = True
                    current_metabolite = {}
                    continue
                
                # Check for metabolite end
                if '</metabolite>' in line:
                    in_metabolite = False
                    # Save if has required fields
                    if ('hmdb_id' in current_metabolite and 
                        'name' in current_metabolite and 
                        'monoisotopic_mass' in current_metabolite and
                        current_metabolite['monoisotopic_mass'] is not None):
                        metabolites.append(current_metabolite.copy())
                        count += 1
                        
                        if max_metabolites and count >= max_metabolites:
                            logger.info(f"Reached limit of {max_metabolites} metabolites")
                            break
                    current_metabolite = {}
                    continue
                
                if not in_metabolite:
                    continue
                
                # Check for taxonomy section
                if '<taxonomy>' in line:
                    in_taxonomy = True
                    continue
                if '</taxonomy>' in line:
                    in_taxonomy = False
                    continue
                
                # Extract fields
                if '<accession>' in line:
                    current_metabolite['hmdb_id'] = self._extract_text(line, 'accession')
                elif '<name>' in line and not in_taxonomy:
                    current_metabolite['name'] = self._extract_text(line, 'name')
                elif '<monoisotopic_molecular_weight>' in line:
                    try:
                        mass_text = self._extract_text(line, 'monoisotopic_molecular_weight')
                        if mass_text:
                            mass = float(mass_text)
                            if mass > 0:
                                current_metabolite['monoisotopic_mass'] = mass
                    except:
                        pass
                elif '<chemical_formula>' in line:
                    current_metabolite['formula'] = self._extract_text(line, 'chemical_formula')
                elif in_taxonomy:
                    if '<kingdom>' in line:
                        current_metabolite['kingdom'] = self._extract_text(line, 'kingdom')
                    elif '<super_class>' in line:
                        current_metabolite['super_class'] = self._extract_text(line, 'super_class')
                    elif '<class>' in line:
                        current_metabolite['class'] = self._extract_text(line, 'class')
            
        except Exception as e:
            logger.error(f"Error parsing HMDB: {e}")
            logger.info(f"Parsed {len(metabolites)} metabolites before error")
        
        finally:
            file_handle.close()
        
        # Convert to DataFrame
        df = pd.DataFrame(metabolites)
        
        # Add average_mass column if not present
        if len(df) > 0 and 'average_mass' not in df.columns:
            df['average_mass'] = df['monoisotopic_mass']
        
        # Add sub_class if not present
        if len(df) > 0 and 'sub_class' not in df.columns:
            df['sub_class'] = None
        
        logger.info(f"✓ Parsed {len(df)} metabolites from HMDB")
        
        self.metabolites = df
        return df
    
    def _extract_text(self, line: str, tag: str) -> Optional[str]:
        """
        Extract text from XML tag in a line.
        
        Args:
            line (str): XML line
            tag (str): Tag name
            
        Returns:
            str: Extracted text or None
        """
        try:
            start_tag = f'<{tag}>'
            end_tag = f'</{tag}>'
            
            if start_tag in line and end_tag in line:
                start_idx = line.find(start_tag) + len(start_tag)
                end_idx = line.find(end_tag)
                text = line[start_idx:end_idx].strip()
                return text if text else None
        except:
            pass
        
        return None
    
    def _extract_metabolite_info(self, elem) -> Optional[Dict]:
        """
        Extract relevant information from a metabolite XML element.
        
        Args:
            elem: XML element
            
        Returns:
            dict: Metabolite information
        """
        try:
            # Define namespace
            ns = {'hmdb': 'http://www.hmdb.ca'}
            
            # Helper function to get text safely
            def get_text(element, tag):
                child = element.find(f'.//hmdb:{tag}', ns)
                if child is not None and child.text:
                    return child.text.strip()
                return None
            
            # Extract basic info
            hmdb_id = get_text(elem, 'accession')
            name = get_text(elem, 'name')
            
            # Skip if no name or ID
            if not hmdb_id or not name:
                return None
            
            # Extract mass info
            monoisotopic_mass = get_text(elem, 'monoisotopic_molecular_weight')
            average_mass = get_text(elem, 'average_molecular_weight')
            
            # Convert to float
            try:
                monoisotopic_mass = float(monoisotopic_mass) if monoisotopic_mass else None
                average_mass = float(average_mass) if average_mass else None
            except:
                monoisotopic_mass = None
                average_mass = None
            
            # Skip if no mass
            if not monoisotopic_mass or monoisotopic_mass <= 0:
                return None
            
            # Extract other info
            formula = get_text(elem, 'chemical_formula')
            
            # Try different taxonomy paths
            taxonomy = elem.find('.//hmdb:taxonomy', ns)
            if taxonomy is not None:
                kingdom = get_text(taxonomy, 'kingdom')
                super_class = get_text(taxonomy, 'super_class')
                class_name = get_text(taxonomy, 'class')
                sub_class = get_text(taxonomy, 'sub_class')
            else:
                kingdom = None
                super_class = None
                class_name = None
                sub_class = None
            
            return {
                'hmdb_id': hmdb_id,
                'name': name,
                'monoisotopic_mass': monoisotopic_mass,
                'average_mass': average_mass,
                'formula': formula,
                'kingdom': kingdom,
                'super_class': super_class,
                'class': class_name,
                'sub_class': sub_class
            }
            
        except Exception as e:
            logger.debug(f"Error extracting metabolite info: {e}")
            return None


class MetaboliteIdentifier:
    """
    Identifies metabolites by matching m/z values to database.
    """
    
    # Common ionization adducts
    ADDUCTS_POSITIVE = {
        '[M+H]+': 1.007276,
        '[M+Na]+': 22.989218,
        '[M+K]+': 38.963158,
        '[M+NH4]+': 18.033823,
        '[M+H-H2O]+': -17.002740,
    }
    
    ADDUCTS_NEGATIVE = {
        '[M-H]-': -1.007276,
        '[M+Cl]-': 34.969402,
        '[M+FA-H]-': 44.998201,
        '[M+Ac-H]-': 59.013851,
    }
    
    def __init__(self, hmdb_database: pd.DataFrame, mass_tolerance_ppm: float = 10.0):
        """
        Initialize metabolite identifier.
        
        Args:
            hmdb_database (DataFrame): Parsed HMDB database
            mass_tolerance_ppm (float): Mass tolerance in ppm
        """
        self.database = hmdb_database
        self.mass_tolerance_ppm = mass_tolerance_ppm
        
        logger.info(f"MetaboliteIdentifier initialized")
        logger.info(f"  Database size: {len(hmdb_database)} metabolites")
        logger.info(f"  Mass tolerance: {mass_tolerance_ppm} ppm")
    
    def calculate_mz_from_mass(self, 
                               neutral_mass: float,
                               adduct: str = '[M+H]+') -> float:
        """
        Calculate m/z from neutral mass and adduct.
        
        Args:
            neutral_mass (float): Neutral molecular mass
            adduct (str): Ionization adduct
            
        Returns:
            float: m/z value
        """
        adduct_mass = self.ADDUCTS_POSITIVE.get(adduct) or self.ADDUCTS_NEGATIVE.get(adduct, 0)
        return neutral_mass + adduct_mass
    
    def search_by_mz(self,
                    mz: float,
                    polarity: str = 'positive',
                    top_n: int = 10) -> pd.DataFrame:
        """
        Search for metabolites matching an m/z value.
        
        Args:
            mz (float): Observed m/z value
            polarity (str): 'positive' or 'negative'
            top_n (int): Number of top matches to return
            
        Returns:
            DataFrame: Matched metabolites with scores
        """
        # Select adducts based on polarity
        if polarity == 'positive':
            adducts = self.ADDUCTS_POSITIVE
        else:
            adducts = self.ADDUCTS_NEGATIVE
        
        matches = []
        
        # Try each adduct
        for adduct_name, adduct_mass in adducts.items():
            # Calculate expected neutral mass
            expected_neutral_mass = mz - adduct_mass
            
            # Calculate mass tolerance in Daltons
            tolerance_da = expected_neutral_mass * self.mass_tolerance_ppm / 1e6
            
            # Find metabolites within tolerance
            mass_diffs = np.abs(self.database['monoisotopic_mass'] - expected_neutral_mass)
            within_tolerance = mass_diffs <= tolerance_da
            
            if within_tolerance.any():
                matching_metabolites = self.database[within_tolerance].copy()
                matching_metabolites['observed_mz'] = mz
                matching_metabolites['adduct'] = adduct_name
                matching_metabolites['mass_error_da'] = (
                    matching_metabolites['monoisotopic_mass'] - expected_neutral_mass
                )
                matching_metabolites['mass_error_ppm'] = (
                    matching_metabolites['mass_error_da'] / expected_neutral_mass * 1e6
                )
                
                matches.append(matching_metabolites)
        
        if not matches:
            return pd.DataFrame()
        
        # Combine all matches
        all_matches = pd.concat(matches, ignore_index=True)
        
        # Sort by absolute mass error
        all_matches['abs_mass_error_ppm'] = np.abs(all_matches['mass_error_ppm'])
        all_matches = all_matches.sort_values('abs_mass_error_ppm')
        
        # Return top N
        return all_matches.head(top_n)
    
    def annotate_feature_list(self,
                             features_df: pd.DataFrame,
                             mz_column: str = 'mz',
                             polarity: str = 'positive',
                             top_n: int = 3) -> pd.DataFrame:
        """
        Annotate a list of features with metabolite identifications.
        
        Args:
            features_df (DataFrame): Feature list with m/z values
            mz_column (str): Column name containing m/z values
            polarity (str): Ionization polarity
            top_n (int): Number of matches per feature
            
        Returns:
            DataFrame: Annotated features
        """
        logger.info(f"Annotating {len(features_df)} features...")
        
        annotations = []
        
        for idx, row in tqdm(features_df.iterrows(), total=len(features_df), desc="Annotating"):
            mz = row[mz_column]
            
            # Search for matches
            matches = self.search_by_mz(mz, polarity=polarity, top_n=top_n)
            
            if len(matches) > 0:
                # Take best match
                best_match = matches.iloc[0]
                
                annotation = {
                    'feature_id': row.get('feature_id', f'F_{mz:.4f}'),
                    'mz': mz,
                    'top_match_name': best_match['name'],
                    'top_match_hmdb_id': best_match['hmdb_id'],
                    'top_match_formula': best_match['formula'],
                    'top_match_adduct': best_match['adduct'],
                    'top_match_mass_error_ppm': best_match['mass_error_ppm'],
                    'n_matches': len(matches),
                    'all_matches': '; '.join(matches['name'].head(3).tolist()),
                    'super_class': best_match['super_class'],
                    'class': best_match['class']
                }
            else:
                annotation = {
                    'feature_id': row.get('feature_id', f'F_{mz:.4f}'),
                    'mz': mz,
                    'top_match_name': 'Unknown',
                    'top_match_hmdb_id': None,
                    'top_match_formula': None,
                    'top_match_adduct': None,
                    'top_match_mass_error_ppm': None,
                    'n_matches': 0,
                    'all_matches': None,
                    'super_class': None,
                    'class': None
                }
            
            annotations.append(annotation)
        
        annotations_df = pd.DataFrame(annotations)
        
        # Merge with original data
        result = features_df.merge(annotations_df, on='feature_id', how='left', suffixes=('', '_annotation'))
        
        # Count identified
        n_identified = (annotations_df['n_matches'] > 0).sum()
        logger.info(f"✓ Identified {n_identified}/{len(features_df)} features ({n_identified/len(features_df)*100:.1f}%)")
        
        return result


def annotate_biomarker_panel(
    biomarker_panel_path: str,
    hmdb_database_path: str,
    output_path: str,
    polarity: str = 'positive',
    mass_tolerance_ppm: float = 10.0,
    max_hmdb_entries: Optional[int] = None
) -> pd.DataFrame:
    """
    Convenience function to annotate a biomarker panel.
    
    Args:
        biomarker_panel_path (str): Path to biomarker panel CSV
        hmdb_database_path (str): Path to HMDB XML file
        output_path (str): Where to save annotated results
        polarity (str): Ionization polarity
        mass_tolerance_ppm (float): Mass tolerance
        max_hmdb_entries (int, optional): Limit HMDB parsing (for testing)
        
    Returns:
        DataFrame: Annotated biomarkers
    """
    # Parse HMDB
    logger.info("Step 1: Parsing HMDB database...")
    parser = HMDBParser()
    hmdb_db = parser.parse_hmdb_xml(hmdb_database_path, max_metabolites=max_hmdb_entries)
    
    if len(hmdb_db) == 0:
        logger.error("Failed to parse HMDB database")
        return pd.DataFrame()
    
    # Load biomarker panel
    logger.info("Step 2: Loading biomarker panel...")
    biomarkers = pd.read_csv(biomarker_panel_path)
    logger.info(f"  Loaded {len(biomarkers)} biomarkers")
    
    # Initialize identifier
    logger.info("Step 3: Initializing metabolite identifier...")
    identifier = MetaboliteIdentifier(hmdb_db, mass_tolerance_ppm=mass_tolerance_ppm)
    
    # Annotate
    logger.info("Step 4: Annotating biomarkers...")
    annotated = identifier.annotate_feature_list(biomarkers, polarity=polarity)
    
    # Save results
    logger.info("Step 5: Saving annotated results...")
    annotated.to_csv(output_path, index=False)
    logger.info(f"✓ Annotated biomarkers saved: {output_path}")
    
    return annotated


# Example usage
if __name__ == "__main__":
    print("Metabolite Identification Module")
    print("=" * 70)
    print()
    print("This module identifies metabolites by matching m/z to HMDB.")
    print()
    print("=" * 70)