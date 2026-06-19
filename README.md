<div align="center">

<h1 align="center">🤖 Machine Learning — Fall 2025</h1>
<h3 align="center">Programming Homework Assignments | Sharif University of Technology</h3>

[**GitHub**](https://github.com/javdaninasim/ML_PHW_Fall_2025) &nbsp; ⬩ &nbsp; [**University**](https://sharif.edu/)

</div>

---

### 📚 Course Overview

```python
class MachineLearning:
    def __init__(self):
        self.institution = "Sharif University of Technology"
        self.instructor = "Ali Sharifi Zarchi"
        self.semester = "Fall 2025"
        self.format = "Jupyter Notebooks"
        
    def topics_covered(self):
        return {
            "Supervised Learning": ["Polynomial Regression", "Perceptron", "Decision Boundaries"],
            "Unsupervised Learning": ["K-Means Clustering", "Gaussian Mixture Models (GMM)", "PCA"],
            "Deep Learning": ["Neural Networks & Optimization", "Convolutional Neural Networks (CNNs)"],
            "Advanced Topics": ["Neural Style Transfer (NST)", "Word Embeddings", "Fine-Tuning"]
        }
    
    def mission(self):
        return "Master machine learning algorithms from foundations to advanced deep learning."
```

---

### 📂 Repository Structure

```
ML_PHW_Fall_2025/
│
├── 📈 Supervised Learning/
│   ├── Polynomial Regression and Perceptron.ipynb
│   │   ├── Part 1: Polynomial Regression (aircraft fuel consumption prediction)
│   │   └── Part 2: Perceptron (Iris dataset classification)
│   │
│   └── Random Forest.ipynb
│       └── Ensemble learning with Random Forests
│
├── 🎯 GMM-KMeans-PCA/
│   ├── K_Means.ipynb                          # K-Means clustering implementation
│   ├── GMM.ipynb                              # Gaussian Mixture Models
│   └── PCA.ipynb                              # Principal Component Analysis for image compression
│
├── 🧠 Neural_Networks_&_Optimization/
│   └── Neural_Networks_&_Optimization.ipynb   # MLPs, activation functions, optimization techniques
│
├── 🎨 Computer Vision/
│   └── CNN & NST.ipynb                        # Convolutional Neural Networks & Neural Style Transfer
│
└── 💬 NLP - Word Embeddings & Fine-Tuning/
    └── ML_HW5_WordEmbeddings_and_FineTuning.ipynb  # Word embeddings and model fine-tuning
```

---

### 🎓 Assignment Breakdown

| # | Topic | Key Concepts | Notebook |
| :---: | :--- | :--- | :--- |
| **1** | **Polynomial Regression** | Bias-Variance Trade-off, Overfitting/Underfitting, Feature Engineering | Supervised Learning |
| **2** | **Perceptron** | Binary Classification, Decision Boundaries, Iris Dataset | Supervised Learning |
| **3** | **Random Forest** | Ensemble Methods, Tree-Based Learning, Feature Importance | Supervised Learning |
| **4a** | **K-Means Clustering** | Unsupervised Learning, Convergence Analysis, Cluster Evaluation | GMM-KMeans-PCA |
| **4b** | **Gaussian Mixture Models** | Probabilistic Clustering, EM Algorithm, Model Selection | GMM-KMeans-PCA |
| **4c** | **PCA** | Dimensionality Reduction, Eigenvalue Decomposition, Image Compression | GMM-KMeans-PCA |
| **5** | **Neural Networks & Optimization** | MLPs, Backpropagation, Activation Functions, Optimizers | Neural_Networks_&_Optimization |
| **6** | **CNNs & Neural Style Transfer** | Convolutional Layers, Feature Extraction, Artistic Style Transfer | Computer Vision |
| **7** | **Word Embeddings & Fine-Tuning** | Word2Vec, Transfer Learning, Pre-trained Models | NLP |

---

### 📝 Detailed Assignment Descriptions

#### **1. Polynomial Regression & Perceptron** 📈

**File:** `Supervised Learning/Polynomial Regression and Perceptron.ipynb`

**Part 1: Polynomial Regression**
- **Problem:** Predict aircraft fuel consumption based on flight speed
- **True Model:** `y = -0.05v⁴ + 2v³ - 3v² + v + 120 + ε` (where ε ~ N(0, 50²))
- **Tasks:**
  - Simulate synthetic data with Gaussian noise
  - Train polynomial models of degrees [1, 2, 3, 5, 20, 30]
  - Analyze bias-variance trade-off
  - Plot MSE vs polynomial degree
  - Identify underfitting and overfitting
- **Output:** Train/test MSE for each degree, visualization of fitted curves

**Part 2: Perceptron**
- **Dataset:** Iris (2 classes: Setosa vs Versicolor, 2 features: Sepal Length & Petal Length)
- **Implementation:** Perceptron from scratch (no scikit-learn)
- **Algorithm:**
  - Net input: `z = w·x + b`
  - Activation: step function (z ≥ 0 → 1, else → -1)
  - Update rule: `w_new = w_old + η(y_i - ŷ_i)x_i`
- **Output:** Learning curve, decision boundary visualization, convergence analysis

**Key Metrics:** MSE, classification accuracy, convergence epochs

---

#### **2. Random Forest** 🌲

**File:** `Supervised Learning/Random Forest.ipynb`

**Concepts Covered:**
- Ensemble learning principles
- Bootstrap aggregating (bagging)
- Random feature selection
- Tree depth and feature importance
- Out-of-bag (OOB) error estimation

---

#### **3. K-Means Clustering** 🎯

**File:** `GMM-KMeans-PCA/K_Means.ipynb`

**Key Topics:**
- K-Means algorithm implementation
- Centroid initialization strategies
- Convergence analysis
- Silhouette score and elbow method
- Clustering visualization

---

#### **4. Gaussian Mixture Models (GMM)** 📊

**File:** `GMM-KMeans-PCA/GMM.ipynb`

**Key Topics:**
- Probabilistic clustering
- Expectation-Maximization (EM) algorithm
- Gaussian component estimation
- Model selection criteria (BIC, AIC)
- Soft clustering vs hard clustering

---

#### **5. Principal Component Analysis (PCA)** 🖼️

**File:** `GMM-KMeans-PCA/PCA.ipynb`

**Problem:** Image compression using dimensionality reduction

**Implementation (From Scratch):**
- Calculate covariance matrix: `Cov = (1/n) X^T X`
- Eigenvalue decomposition: `eigvals, eigvecs = np.linalg.eigh(Cov)`
- Select top k components
- Project: `X_compressed = X · W_k`
- Reconstruct: `X_reconstructed = X_compressed · W_k^T + mean`

**Pipeline:**
1. **Image Preparation:** Load grayscale image
2. **Patch Extraction:** Divide into 12×12 patches → (N, 144) matrix
3. **PCA Training:** Learn 16 principal components
4. **Compression:** Project to 16-dimensional space
5. **Reconstruction:** Inverse transform back to 144 dimensions
6. **Stitching:** Reassemble patches into full image
7. **Visualization:** Show top 16 eigenpatches

**Output:** 
- Reconstruction MSE
- Original vs reconstructed image comparison
- Top 16 eigenpatches visualization
- Compression ratio: 144 → 16 (9x reduction)

---

#### **6. Neural Networks & Optimization** 🧠

**File:** `Neural_Networks_&_Optimization/Neural_Networks_&_Optimization.ipynb`

**Concepts:**
- Multi-layer perceptron (MLP) architecture
- Activation functions (ReLU, Sigmoid, Tanh)
- Backpropagation algorithm
- Gradient descent variants (SGD, Adam, RMSprop)
- Batch normalization and dropout
- Hyperparameter tuning
- Learning rate scheduling

---

#### **7. Convolutional Neural Networks & Neural Style Transfer** 🎨

**File:** `Computer Vision/CNN & NST.ipynb`

**Topics:**
- Convolutional layers and filters
- Pooling and feature maps
- Image classification with CNNs
- Pretrained models (VGG, ResNet)
- Neural Style Transfer algorithm
- Content and style loss functions
- Artistic rendering

---

#### **8. Word Embeddings & Fine-Tuning** 💬

**File:** `NLP - Word Embeddings & Fine-Tuning/ML_HW5_WordEmbeddings_and_FineTuning.ipynb`

**Topics:**
- Word2Vec (Skip-gram, CBOW)
- Word embeddings visualization
- Semantic similarity
- Transfer learning
- Fine-tuning pretrained language models
- Text classification with embeddings

---

### ⚡ Tech Stack

<div align="left">
  <img src="https://img.shields.io/badge/Python-1A1A1A?style=for-the-badge&logo=python&logoColor=3776AB" alt="Python" />
  <img src="https://img.shields.io/badge/Jupyter-1A1A1A?style=for-the-badge&logo=jupyter&logoColor=F37726" alt="Jupyter" />
  <img src="https://img.shields.io/badge/NumPy-1A1A1A?style=for-the-badge&logo=numpy&logoColor=013243" alt="NumPy" />
  <img src="https://img.shields.io/badge/Pandas-1A1A1A?style=for-the-badge&logo=pandas&logoColor=150458" alt="Pandas" />
  <img src="https://img.shields.io/badge/scikit--learn-1A1A1A?style=for-the-badge&logo=scikit-learn&logoColor=F7931E" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/PyTorch-1A1A1A?style=for-the-badge&logo=pytorch&logoColor=EE4C2C" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Matplotlib-1A1A1A?style=for-the-badge&logo=python&logoColor=11557C" alt="Matplotlib" />
  <img src="https://img.shields.io/badge/Seaborn-1A1A1A?style=for-the-badge&logo=python&logoColor=0055D4" alt="Seaborn" />
</div>

<br>

> **Core Libraries:** `NumPy` ⬩ `Pandas` ⬩ `scikit-learn` ⬩ `PyTorch` ⬩ `Matplotlib` ⬩ `Seaborn`

---

### 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/javdaninasim/ML_PHW_Fall_2025.git
cd ML_PHW_Fall_2025

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install numpy pandas scikit-learn torch matplotlib seaborn jupyter pillow

# 4. Launch Jupyter and explore assignments
jupyter notebook
```

---

### 📖 How to Use This Repository

Each folder contains interactive **Jupyter Notebooks** with:

- 📝 **Problem Statements** – Clear problem definitions and objectives
- 🔧 **Code Templates** – Partially completed implementations with TODO comments
- 📊 **Visualizations** – Plots, graphs, and visual analysis
- 💡 **Insights** – Interpretations and key takeaways
- 📚 **References** – Links to papers and external resources

Simply open any `.ipynb` file in Jupyter to start learning!

---

### 💡 Key Learnings

| Algorithm | Application | Status |
| :--- | :--- | :---: |
| **Polynomial Regression** | Curve fitting, bias-variance analysis | ✅ Mastered |
| **Perceptron** | Linear binary classification | ✅ Mastered |
| **Random Forest** | Ensemble learning, feature selection | ✅ Mastered |
| **K-Means** | Unsupervised clustering | ✅ Mastered |
| **GMM** | Probabilistic clustering | ✅ Mastered |
| **PCA** | Dimensionality reduction, compression | ✅ Mastered |
| **Neural Networks** | Deep learning fundamentals | ✅ Mastered |
| **CNNs** | Image recognition and processing | ✅ Mastered |
| **NST** | Artistic style transfer | ✅ Mastered |
| **Word Embeddings** | NLP and transfer learning | ✅ Mastered |

---

### 🎯 Learning Objectives

- ✨ Understand **fundamental ML algorithms** from theory to implementation
- 🧠 Implement **classical ML models from scratch** (Perceptron, PCA, etc.)
- 📊 Apply **ensemble methods** and advanced techniques
- 🤖 Master **deep learning** with PyTorch
- 🎨 Explore **advanced applications** (style transfer, embeddings, fine-tuning)
- 📈 Analyze **model performance** and trade-offs (bias-variance, accuracy-complexity)
- 💪 Develop **production-ready implementations** with proper evaluation

---

### ℹ️ Course Information

| Detail | Information |
| :--- | :--- |
| **Institution** | Sharif University of Technology, Dept. of Computer Engineering |
| **Course Code** | CE 40635 |
| **Instructor** | Dr. Ali Sharifi Zarchi |
| **Semester** | Fall 2025 (Mehr–Aban 1404) |
| **Format** | Interactive Jupyter Notebooks |
| **Prerequisites** | Linear Algebra, Calculus, Statistics, Python |
| **Languages** | Python, Jupyter Notebook |

---

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=000000&height=100&section=footer" width="100%"/>
</div>
