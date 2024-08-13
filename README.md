# Disentangling-cyber-risk-premia

## Goal

This study aims to build various cyber risk factors based on the 10-K disclosures of public firms and show that those factors are robust to all factors' benchmarks. The study is an extension of the work performed by Daniel Celeny and Loïc Maréchal: https://github.com/technometrics-lab/17-Cyber-risk_and_the_cross-section_of_stock_returns





## Summary of the paper

The main focus of this paper is to disentangle the different cyber risks faced by firms and the effects on expected returns through risk premia channels. To do this, I collect financial fillings, monthly returns, and other firm characteristics for over 7000 firms listed on US stock markets between January 2007 and December 2023. I use a neural network called "Paragraph Vector" in combination with the MITRE ATT&CK cybersecurity knowledgebase and clustering techniques to score each firm's filing based on its various types of cyber risk.

I find four types of cyber attacks emerge from the textual cluster structures of MITRE ATT&CK. I establish scores to quantify the similarity between the annual statements of firms, the 10-Ks, and the identified types of cyber attacks from the knowledgebase MITRE ATT&CK. I find that the four "cyber" scores present no correlation with standard firms' characteristics known to help price stock returns and weak correlations with textual non-semantic variables of the annual statement (the highest, $0.36$, correlates with the length of section 1.A., in the 10-Ks). As previously observed in a study using the same neural network, the resulting aggregation of the various cyber scores shows increasing trends, with scores increasing by $0.04$ from 2007 to 2023. I also find that specific industries from the Fama-French 12-industries classification display higher cyber scores, with Business Equipment and Telephone and Television Transmission being the highest.

I find that organizing firms into portfolios based on their cyber scores, with increasing cyber scores, results in progressively higher average excess returns. All average excess returns of all portfolios are statistically significant at the 1% level, and investing in a portfolio that enters a long (short) position in the top (bottom) cyber scores firm is statistically significant at the 5% level. The aforementioned results stay true at the 5% and 10% levels in the top portfolios after controlling for common risk factors. 

The risk premia associated with the different types of cyber risk are also manifest at the 5% level in the cross-section with Fama and MacBeth (1973) regressions. Using additional pricing factors related to cyber-based portfolios improves pricing ability. I demonstrate that joint alphas of various assets tend to decrease in Gibbons, Ross and Shanken (1989) tests. Using the Bayesian approach of Barillas and Shanken (2018), I also demonstrate that the optimal subset of factors pricing stock returns invariably includes the cyber-based factors.

Additional tests reveal that, although various types of cyber risk exist, the market does not differentiate between them and perceives them as a single aggregate cyber risk. Finally, I conduct an event analysis to evaluate the performance of a cyber-based portfolio during the massive SolarWinds cyber attack in December 2020. Contrary to previous studies, no significant conclusions can be drawn from this event regarding the performance of my cyber-based portfolios in cyber-related crises.



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
