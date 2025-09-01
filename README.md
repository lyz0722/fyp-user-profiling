1.Project Structure

- `main2.py` – **Main script** to run the full pipeline (preprocessing, clustering, time-series analysis, recommendation).
- `preprocessing.py` – Data cleaning and transformation functions.
- `eda.py` – Exploratory Data Analysis utilities.
- `feature_engineering.py` – Feature extraction from user behavior logs.
- `clustering.py` – Clustering algorithms (K-Means, GMM, etc.).
- `q_learning.py` – Reinforcement learning module for action recommendation.
- Other utility scripts as needed.

2.Dataset

Due to the large size of the dataset (~1.5GB), it is not included in this GitHub repository.

However, a direct download link to the dataset (hosted externally) will be provided in the appendix of the submitted thesis.

The dataset is based on the Alibaba CIKM 2019 e-commerce behavior logs.

3.How to Run

All scripts were developed and tested in **VS Code** using Python 3.8+ in a local Anaconda environment.

To execute the main analysis pipeline:

```bash
python main2.py


