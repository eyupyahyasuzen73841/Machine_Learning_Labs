# Domain-Driven Problem Formulation: Income Stratification

**Student:** Eyüp Yahya Süzen (ID: 73841)  
**Cohort:** 1994 United States Census Population Survey  

### 1. Architectural Scope
The objective is the construction of a resilient, non-leaking data pipeline to identify survey respondents whose earnings exceed $50,000 annually.

### 2. Predictor Attributes & Classification Target
* **Predictor Attributes (14 Dimensions):** 6 numerical columns (`age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`) and 8 discrete factors (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`).
* **Target:** Binary indicator (0, 1) where 1 represents `>50K`.

### 3. Evaluation Metrics
Class distribution is asymmetric (~76% <=50K). Accuracy is insufficient; **F1-Score** is the central success criterion to penalize false negatives in high-earner identification.
