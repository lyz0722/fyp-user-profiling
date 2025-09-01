1.Project structure

- `main2.py` – Main script that executes the complete pipeline, including preprocessing, clustering, time-series modeling, and recommendations.
- `preprocessing.py` – Handles data cleaning and transformation from raw behavior logs.
- `eda.py` – Performs exploratory data analysis and summary statistics.
- `feature_engineering.py` – Extracts user-level and item-level features for modeling.
- `clustering.py` – Contains clustering algorithms such as K-Means, DBSCAN, and GMM.
- `timeseries_modeling.py` – Models temporal behavior trends based on relative days.
- `recommendation.py` – Core recommendation logic and policy definitions.
- `recommendation_strategies.py` – Predefined recommendation strategies (e.g., recommend_A/B/C).
- `evaluation.py` – Evaluation functions for clustering quality and data profiling.


2.Dataset

Due to the large size of the dataset (~1.5GB), it is not included in this GitHub repository.

However, a direct download link to the dataset (hosted externally) will be provided in the appendix of the submitted thesis.

The dataset is based on the Alibaba CIKM 2019 e-commerce behavior logs.

3.How to Run

All scripts were developed and tested in **VS Code** using Python 3.8+ in a local Anaconda environment.

To execute the main analysis pipeline:

```bash
python main2.py


