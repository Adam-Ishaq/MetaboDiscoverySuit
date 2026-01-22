"""
Run Metabolite Identification on Biomarker Panel
Matches m/z values to HMDB database to identify metabolites.
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.metabolite_identification.metabolite_identification import (
    HMDBParser, MetaboliteIdentifier
)


def main():
    """Main metabolite identification execution."""
    
    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║       Metabolite Identification - Malaria Biomarkers              ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Configuration
    config = {
        'hmdb_path': 'data/databases/hmdb_metabolites.xml',  # Adjust if needed
        'biomarker_panel': 'results/ml_biomarker_discovery/random_forest_biomarker_panel.csv',
        'statistical_results': 'results/statistical_analysis/top50_features.csv',
        'output_dir': 'results/metabolite_identification',
        'polarity': 'positive',  # or 'negative' based on your data
        'mass_tolerance_ppm': 10.0,
        'max_hmdb_entries': None,  # Set to 10000 for testing, None for full database
    }
    
    print("Configuration:")
    print("-" * 70)
    for key, value in config.items():
        print(f"  {key:25s}: {value}")
    print()
    
    # Check if HMDB file exists
    hmdb_path = Path(config['hmdb_path'])
    if not hmdb_path.exists():
        print("❌ HMDB database file not found!")
        print(f"   Expected location: {hmdb_path}")
        print()
        print("Please:")
        print("  1. Download HMDB XML from: https://hmdb.ca/downloads")
        print("  2. Place it in: data/databases/")
        print("  3. Update config['hmdb_path'] if needed")
        print()
        return
    
    # Parse HMDB
    print("=" * 70)
    print("Step 1: Parsing HMDB Database")
    print("=" * 70)
    print("⚠ Note: Full HMDB parsing takes 10-30 minutes!")
    print("   For testing, set max_hmdb_entries=10000 in config")
    print()
    
    parser = HMDBParser()
    hmdb_db = parser.parse_hmdb_xml(
        str(hmdb_path),
        max_metabolites=config['max_hmdb_entries']
    )
    
    if len(hmdb_db) == 0:
        print("❌ Failed to parse HMDB database")
        return
    
    print()
    print(f"Database Statistics:")
    print(f"  Total metabolites: {len(hmdb_db)}")
    print(f"  Mass range: {hmdb_db['monoisotopic_mass'].min():.2f} - {hmdb_db['monoisotopic_mass'].max():.2f} Da")
    print()
    
    # Initialize identifier
    print("=" * 70)
    print("Step 2: Initializing Metabolite Identifier")
    print("=" * 70)
    
    identifier = MetaboliteIdentifier(
        hmdb_db,
        mass_tolerance_ppm=config['mass_tolerance_ppm']
    )
    print()
    
    # Annotate ML biomarker panel
    print("=" * 70)
    print("Step 3: Annotating ML Biomarker Panel")
    print("=" * 70)
    
    biomarker_path = Path(config['biomarker_panel'])
    if biomarker_path.exists():
        biomarkers = pd.read_csv(biomarker_path)
        print(f"Loaded {len(biomarkers)} biomarkers from ML analysis")
        
        # Add feature_id if not present
        if 'feature_id' not in biomarkers.columns and 'feature' in biomarkers.columns:
            biomarkers['feature_id'] = biomarkers['feature']
        
        # Extract m/z from feature_id if needed (format: F_293.91_1.64)
        if 'mz' not in biomarkers.columns:
            biomarkers['mz'] = biomarkers['feature_id'].str.extract(r'F_(\d+\.\d+)_')[0].astype(float)
        
        annotated_ml = identifier.annotate_feature_list(
            biomarkers,
            mz_column='mz',
            polarity=config['polarity'],
            top_n=3
        )
        
        # Save
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "annotated_ml_biomarkers.csv"
        annotated_ml.to_csv(output_path, index=False)
        print(f"✓ Saved: {output_path}")
        
        # Print summary
        print()
        print("Top 10 Identified Biomarkers:")
        print("-" * 70)
        for i, row in annotated_ml.head(10).iterrows():
            if row.get('top_match_name') != 'Unknown':
                print(f"{i+1:2d}. m/z {row['mz']:.4f} → {row['top_match_name']}")
                print(f"    HMDB: {row['top_match_hmdb_id']}, Formula: {row['top_match_formula']}")
                print(f"    Adduct: {row['top_match_adduct']}, Error: {row['top_match_mass_error_ppm']:.2f} ppm")
            else:
                print(f"{i+1:2d}. m/z {row['mz']:.4f} → Unknown (no match)")
            print()
    
    else:
        print(f"⚠ Biomarker panel not found: {biomarker_path}")
        print("  Run ML analysis first!")
    
    print()
    
    # Annotate statistical top features
    print("=" * 70)
    print("Step 4: Annotating Statistical Top Features")
    print("=" * 70)
    
    stats_path = Path(config['statistical_results'])
    if stats_path.exists():
        top_features = pd.read_csv(stats_path)
        print(f"Loaded {len(top_features)} top statistical features")
        
        # Add mz column if not present
        if 'mz' not in top_features.columns:
            # Try to extract from feature_id
            if 'feature_id' in top_features.columns:
                top_features['mz'] = top_features['feature_id'].str.extract(r'F_(\d+\.\d+)_')[0].astype(float)
        
        annotated_stats = identifier.annotate_feature_list(
            top_features,
            mz_column='mz',
            polarity=config['polarity'],
            top_n=3
        )
        
        # Save
        output_path = output_dir / "annotated_statistical_features.csv"
        annotated_stats.to_csv(output_path, index=False)
        print(f"✓ Saved: {output_path}")
        
        # Count identification rate
        identified = (annotated_stats['n_matches'] > 0).sum()
        print(f"✓ Identified {identified}/{len(annotated_stats)} features ({identified/len(annotated_stats)*100:.1f}%)")
    
    else:
        print(f"⚠ Statistical results not found: {stats_path}")
    
    print()
    
    # Create summary
    print("=" * 70)
    print("Step 5: Creating Summary")
    print("=" * 70)
    
    if biomarker_path.exists() and stats_path.exists():
        # Combine and create final biomarker list
        final_biomarkers = annotated_ml[annotated_ml['n_matches'] > 0].copy()
        
        # Add biological classification summary
        if len(final_biomarkers) > 0:
            print("\nBiomarker Classification Summary:")
            print("-" * 70)
            
            if 'super_class' in final_biomarkers.columns:
                class_counts = final_biomarkers['super_class'].value_counts()
                for class_name, count in class_counts.head(10).items():
                    print(f"  {class_name:30s}: {count} biomarkers")
            
            # Save final list
            final_path = output_dir / "final_annotated_biomarkers.csv"
            final_biomarkers.to_csv(final_path, index=False)
            print(f"\n✓ Final annotated biomarkers saved: {final_path}")
    
    print()
    print("=" * 70)
    print("✓ Metabolite Identification Complete!")
    print("=" * 70)
    print()
    print("Files created:")
    print(f"  - {config['output_dir']}/annotated_ml_biomarkers.csv")
    print(f"  - {config['output_dir']}/annotated_statistical_features.csv")
    print(f"  - {config['output_dir']}/final_annotated_biomarkers.csv")
    print()
    print("Next steps:")
    print("  1. Review annotated biomarkers")
    print("  2. Verify identifications manually")
    print("  3. Look up biomarkers in literature")
    print("  4. Consider MS/MS confirmation")
    print()
    print("🎉 You now have named metabolite biomarkers for malaria! 🦟")
    print("=" * 70)


if __name__ == "__main__":
    main()