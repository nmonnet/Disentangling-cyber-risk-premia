# Disentangling-cyber-risk-premia






| File Name                                       | Description                                                                                                                                                           |
|-------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `similitude_matrix`                             | MITRES matrix auto cosine similitude matrix. Used in clustering program to avoid losing deterministic randomness by calling `useful_function.py`.                      |
| `super_tactics_clustering.ipynb`                | Contains clustering method to find relevant group of "super tactic". Self-sufficient, meaning it does not need outside `.py` file to function.                        |
| `test.ipynb`                                    | Contains a bunch of tests. The code is messy and unordered; don't bother with it.                                                                                     |
| `data_control_and_extract_DC.ipynb`             | Tests to see if the saved doc2vec models produce similar vectors to the ones saved by D.C.                                                                            |
| `density_separated_cyber_risk.ipynb`            | Creates separated cyber signals from cosine similitude matrix (currently based on D.C. cosine similitude matrix).                                                     |
| `10k_similitude_matrix.ipynb`                   | Creates the cosine similitude matrix from tokens of 10-K.                                                                                                             |
| `10k_setup.ipynb`                               | Downloads and tokenizes 10-K, then saves them by years.                                                                                                               |
| `useful_functions.py`                           | Contains all functions used throughout the project. Calling it is an easy way to set up all necessary libraries and functions from D.C.                               |
| `similitude_matrix_setup_and_save.ipynb`        | Creates the similitude matrix (MITRES matrix auto cosine similitude matrix).                                                                                          |
| `financial_variable_extractor.ipynb`            | Extracts all relevant firm financial values from CRSP and WRDS (in `Financial_ratios`) and all returns and prices (in `Financial_values`).                            |
| `bid_ask_pipeline.ipynb`                        | Computes the bid-ask spread when data is available using `ardiaGuidottiKroencke2023W` (not used in the paper).                                                        |
| `bid_ask_to_dataframe_converter`                | Takes each firm-specific bid-ask dataframe obtained through `bid_ask_pipeline.ipynb` and converts it to a single common dataframe (not used in the paper).            |
| `camembert.ipynb`                               | Creates a pie chart of industry proportions (12 classes).                                                                                                             |
| `cyber_histogram_yearly.ipynb`                  | Plots the evolution of cyberscores (histogram through firms and average trend) over time.                                                                             |
| `cyber_table_constructor.ipynb`                 | Takes each firm-specific cyberscore dataframe and converts it to a single common dataframe for all firms.                                                             |
| `empty_price_table.txt`                         | Contains firms for which prices were not available (not very useful).                                                                                                 |
| `financial_textual_variable_extractor.ipynb`    | Creates variables such as secret, readability, risk length from the 10-K.                                                                                             |
| `financial_textual_variable_dataframe_converter`| Takes each firm-specific secret, readability, and risk length dataframe and converts it to a single common dataframe for all firms.                                   |
| `financial_variable_extractor.ipynb`            | Extracts financial variables to test the novelty of the cyberscore later.                                                                                             |
| `pipeline_v0.ipynb`                             | Version 0, might be removed later. Performs early statistical tests regarding the cyberscores.                                                                        |
| `price_bid_ask_estimator_test.ipynb`            | Tests the estimator used to compute the bid-ask spread (not used in the paper).                                                                                       |
| `returns_format.ipynb`                          | Takes each firm-specific returns dataframe and converts it to a single common dataframe for all firms.                                                                |
| `sentiment_analyzer.ipynb`                      | Creates the cyber-sentiment score.                                                                                                                                    |
| `additional_test_paper.ipynb`                   | Additional tests to be included in the last section of the paper (differences of cyberfolios and analysis of cyber attack events).                                     |
| `comparison_P1_P5.ipynb`                        | Old code, irrelevant for the papers.                                                                                                                                  |
| `daily_return_aquisition_for_event_analysis.ipynb`| Extracts and stores daily returns of stocks in a dataframe for use in event analysis.                                                                                  |
| `lecteur_csv.ipynb`                             | Reads the results of the pipeline and formats the aesthetic to be easily translated into a LaTeX table.                                                               |
| `pipeline_v2.ipynb`                             | Pipeline to analyze each cyber score found with the clustering technique.                                                                                             |
| `return_format.ipynb`                           | Reforms monthly returns to make them more suitable for use in the pipeline.                                                                                           |
| `stats_text_df.ipynb`                           | Statistical assessment of the paragraphs identified in the 10-Ks.                                                                                                     |
| `super_tactics_clustering-density_trial.ipynb`  | Code to be used in another paper (ignore it).                                                                                                                         |
| `test_BGRS.ipynb`                               | Test for the BGRS.                                                                                                                                                    |
| `test_BGRS-Copy1.ipynb`                         | Test for the BGRS.                                                                                                                                                    |







Note: In the file data, you will also find output of my work:

10k_statements_new : contain the tokenized 10k and the saved cosine matrix
	-item_1A_limits (currently empty, might be deleted later)
	-paragraph_vectors (currently empty, might be deleted later)
	-similarity_with_MITRES (contain cosine similitude matrix between 10k and MITRES)
	-tokens (tokenized and saved 10k)

stocknames.csv : VERY IMPORTANT FILE, can be generated by launching the first 18 lines of Project_data_acquisition.ipynb (do it before launching something else, it's step 1)
