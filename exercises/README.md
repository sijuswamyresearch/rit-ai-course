# Course Exercises

This directory contains Python implementations for all the exercises mentioned in the course.

## Exercise Files

### Module 1: Mathematical Foundations

1. **exercise_1_1_gradient_descent.py**
   - Implements gradient descent from scratch
   - Visualizes optimization paths for different learning rates
   - Analyzes convergence behavior

2. **exercise_1_2_linear_algebra.py**
   - Matrix operations without NumPy
   - Eigenvalue computation using power iteration
   - SVD for dimensionality reduction
   - PCA from scratch

### Module 2: Classical Machine Learning

3. **exercise_2_1_ml_pipeline.py**
   - Complete ML pipeline from data cleaning to evaluation
   - Multiple model training and comparison
   - Feature engineering and selection

4. **exercise_2_2_clustering.py**
   - K-Means clustering from scratch
   - Hierarchical clustering
   - PCA visualization
   - Cluster analysis

### Module 3: Deep Learning

5. **exercise_3_1_neural_network_scratch.py**
   - Neural network implementation without high-level frameworks
   - Forward and backward propagation
   - Multiple activation functions
   - Decision boundary visualization

6. **exercise_3_2_cnn_image_classification.py**
   - CNN for image classification
   - Data augmentation
   - Model training and evaluation
   - Misclassification analysis

### Module 4: NLP & Generative AI

7. **exercise_4_1_sentiment_analysis.py**
   - Text classification system
   - Text preprocessing
   - Feature extraction (TF-IDF, embeddings)
   - Model evaluation

8. **exercise_4_2_rag_system.py**
   - Retrieval-Augmented Generation system
   - Document indexing
   - Semantic search
   - Answer generation

### Module 5: Deployment

9. **exercise_5_1_streamlit_app.py**
   - Streamlit web application
   - Model deployment
   - Interactive interface

## How to Run

### Prerequisites

Install required packages:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow torch transformers streamlit sentence-transformers
```

### Running Individual Exercises

```bash
# Module 1
python exercises/exercise_1_1_gradient_descent.py
python exercises/exercise_1_2_linear_algebra.py

# Module 2
python exercises/exercise_2_1_ml_pipeline.py
python exercises/exercise_2_2_clustering.py

# Module 3
python exercises/exercise_3_1_neural_network_scratch.py
python exercises/exercise_3_2_cnn_image_classification.py

# Module 4
python exercises/exercise_4_1_sentiment_analysis.py
python exercises/exercise_4_2_rag_system.py

# Module 5
streamlit run exercises/exercise_5_1_streamlit_app.py
```

## Expected Outputs

Each exercise will:
- Print results and metrics to the console
- Generate visualization plots
- Save outputs to the current directory

## Submission Guidelines

For each exercise:
1. Run the code and verify it works
2. Modify and experiment with different parameters
3. Write a brief report on your observations
4. Submit the report along with any modifications you made

## Additional Resources

- Course notes: `modules/module*_notes.qmd`
- Slide decks: `modules/module*.qmd`
- Project guide: `assessments/project_guide.qmd`