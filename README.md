# MetaboDiscoverySuit <img width="100" height="100" alt="Logo" src="https://github.com/user-attachments/assets/f904a723-688b-4398-942e-d33f98ef76b3" />
An AI-powered Python platform for LC-MS metabolomics biomarker discovery with advanced machine learning, automated statistical analysis, and explainable AI capabilities.

<img width="2816" height="1372" alt="MetaboDiscoverSuit" src="https://github.com/user-attachments/assets/4b69c1cb-37cd-4ddc-9247-1a44c569927f" />

**MetaboDiscoverySuit** is a comprehensive, open-source Python platform designed to revolutionize metabolomics biomarker discovery through the power of artificial intelligence and machine learning. Built as an advanced alternative to traditional tools like MetaboAnalyst, this system provides researchers with an end-to-end pipeline for analyzing LC-MS metabolomics data with cutting-edge AI capabilities.

# 🎯 What Makes This Different?
Unlike traditional metabolomics platforms that rely solely on classical statistics, MetaboAI integrates state-of-the-art machine learning algorithms including Random Forests, XGBoost, and Deep Neural Networks to identify biomarkers with unprecedented accuracy. The platform features explainable AI through SHAP values, allowing researchers to understand not just which metabolites are important, but why they matter.

# 🚀 Key Capabilities

- **Complete Data Pipeline:** Process raw mzML files through peak detection, alignment, and feature extraction with optimized algorithms for high-throughput analysis.

- **Advanced Statistics:** Perform univariate tests, PCA, PLS-DA, and multivariate analyses with automated multiple testing correction and comprehensive quality control.

- **AI-Powered Discovery:** Leverage ensemble machine learning models that combine multiple algorithms to identify robust biomarkers, validated through rigorous cross-validation and performance metrics.

- **Metabolite Identification:** Integrate with major databases (HMDB, METLIN, KEGG) for confident metabolite annotation and pathway enrichment analysis.

- **Explainable Results:** Understand model decisions through SHAP values and feature importance rankings, making AI predictions transparent and scientifically interpretable.

<img width="100" height="100" alt="image-removebg-preview" src="https://github.com/user-attachments/assets/8b64dcd6-f532-49d4-97fd-ad89090d3ee1" /> **AI & Machine Learning**

- **Ensemble Learning:** Advanced ensemble methods with Bayesian hyperparameter optimization

- **Deep Learning:** Custom neural architectures (autoencoders, VAEs, transformers)

- **Feature Engineering:** Automated feature selection with ensemble voting

- **Model Interpretation:** SHAP-based explainability and feature importance analysis

- ✅ Random Forest with feature importance

- ✅ XGBoost/LightGBM gradient boosting

- ✅ Deep Neural Networks

- ✅ Autoencoders for feature extraction

- ✅ Ensemble methods

- ✅ SHAP values for explainable AI

- ✅ Cross-validation and model validation

 **Neural Networks for MetaboDiscoverySuit**

- **Autoencoder Networks:** For dimensionality reduction and anomaly detection

- **Variational Autoencoders:** For probabilistic modeling with uncertainty quantification

- **Graph Neural Networks:** For metabolic network analysis

- **Transformer Models:** For sequence and time-series metabolomics data

🆕 **Kolmogorov-Arnold Networks (KANs):** Interpretable networks with learnable activation functions

🆕 **Variational KANs:** Probabilistic KANs with uncertainty quantification

🆕 **Hybrid KANs:** Combining KAN feature extraction with traditional classifiers

**Metabolite Identification**

- ✅ Database integration (HMDB, METLIN)
  
- ✅ m/z matching with mass tolerance

- ✅ Adduct and isotope pattern consideration

- ✅ Confidence scoring

**Visualization**

- ✅ PCA/PLS-DA score plots

- ✅ Heatmaps with hierarchical clustering

- ✅ Volcano plots

- ✅ ROC curves

- ✅ Feature importance plots

- ✅ SHAP summary plots

**Installation Requirements - MetaboDiscoverySuit**

**System Requirements**

- **Minimum Requirements**

- **Operating System:** Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)

- **Python:** 3.9 or higher 

- **RAM:** 8 GB minimum

- **Storage:** 10 GB free space (more for large datasets)

- **Processor:** Multi-core processor (4+ cores recommended)

**Recommended Requirements**

- **RAM:** 16 GB or more (32 GB for large datasets >1000 samples)

- **Storage:** 50 GB+ SSD for better I/O performance

- **Processor:** 8+ cores for parallel processing

- **GPU:** NVIDIA GPU with CUDA support (optional, for deep learning acceleration)

## 🛠️ **Installation**

``` bash
Step 1: Clone the Repository
bashgit clone https://github.com/yourusername/metabolomics-ai-biomarker-discovery.git
cd metabolomics-ai-biomarker-discovery

## **Step 2: Create Virtual Environment**
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n metabolomics python=3.9
conda activate metabolomics
