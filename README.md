# Disentangling-cyber-risk-premia






similitude_matrix 				: MITRES matrix auto cosine similitude matrix. Used in clustering program to avoid loosing deterministic randomness by calling useful_function.py

super_tactics_clustering.ipynb 			: contain clustering method to find relevant group of "super tactic". (self sufficient, meaning it does not need outside .py file to function)

test.ipynb 					: contain a bunch of test, the code is messy and unordered, don't bother with that

data_control_and_extract_DC.ipynb		: test to see if the saved doc2vec models produce similar vectors that the one saved by D.C.

density_separated_cyber_risk.ipynb		: create the separated cyber signals from cosine similitude matrix (currently on the basis of D.C. cosine similitude matrix)

10k_similitude_matrix.ipynb			: create the cosine similitude matrix from tokens of 10k

10k_setup.ipynb					: download and tokenize 10k, then save them by years

useful_functions.py				: contain all the function used throughout the project, calling it is an easy way to set up all the library needed and the functions from D.C.

similitude_matrix_setup_and_save.ipynb		: create the similitude matrix (MITRES matrix auto cosine similitude matrix)

financial_variable_extractor.ipynb		: extract from crsp and wrds all relevant firm financial values (in Financial_ratios) and all returns and prices (in Financial_values)

bid_ask_pipeline.ipynb				: compute the bid-ask spread when data are available using ardiaGuidottiKroencke2023W (not used in the paper)

bid_ask_to_dataframe_converter			: take each firm specific bid-ask dataframe obtain though bid_ask_pipeline.ipynb and convert it to a single common dataframe (not used in the paper)

camembert.ipynb					: create the pie chart of proportion of industry (12 classes)

cyber_histogram_yearly.ipynb			: plot evolution of cyberscores (histogram through firms and average trend) though time 

cyber_table_constructor.ipynb			: take each firm specific cyberscore dataframe and convert it to single common dataframes for all firms

empty_price_table.txt				: contains firms for which price were not available (not very useful)

financial_textual_variable_extractor.ipynb	: create, from the 10k the variable secret, readability risk length 

financial_textual_variable_dataframe_converter.	: take each firm specific secret, readability and risk length dataframe and convert it to single common dataframes for all firms

financial_variable_extractor.ipynb		: extract financial variable to test later the novelty of the cyberscore 

pipeline_v0.ipynb				: version 0, might be removed later, perform early statistical test regarding the cyberscores 

price_bid_ask_estimator_test.ipynb		: test the estimator used to compute the bid-ask spread (not used in the paper)

price_bid_ask_estimator_test.ipynb		: test the estimator used to compute the bid-ask spread (more tests) (not used in the paper)

returns_format.ipynb				: take each firm specific returns dataframe and convert it to a single common dataframe for all firms

sentiment_analyzer.ipynb			: create the cyber-sentiment-score

additional_test_paper.ipynb			: additional tests to be included in the last section of the paper (differences of cyberfolios and analysis of event of cyber attack)

comparison_P1_P5.ipynb				: old code, irrelevant for the papers 

daily_return_aquisition_for_event_analysis.ipynb: extract and stock daily returns of stock in a dataframe to be used in the event analysis

lecteur_csv.ipynb 				: read the results of the pipeline and transform the esthetic to be easily translated into latex table. 

pipeline_v2.ipynb				: pipeline to analyse each cyber score found with the clustering technique

return_format.ipynb				: reform monthly return to make them more suitable to be used in the pipeline

sentiment_analyzer.ipynb			: create the cyber sentiment score

stats_text_df.ipynb				: statistical assessment of the paragraph identified in the 10-Ks

super_tactics_clustering-density_trial.ipynb	: code to be used in another paper (ignore it)

test_BGRS.ipynb					: test for the BGRS 

test_BGRS-Copy1.ipynb				: test for the BGRS 






Note: In the file data, you will also find output of my work:

10k_statements_new : contain the tokenized 10k and the saved cosine matrix
	-item_1A_limits (currently empty, might be deleted later)
	-paragraph_vectors (currently empty, might be deleted later)
	-similarity_with_MITRES (contain cosine similitude matrix between 10k and MITRES)
	-tokens (tokenized and saved 10k)

stocknames.csv : VERY IMPORTANT FILE, can be generated by launching the first 18 lines of Project_data_acquisition.ipynb (do it before launching something else, it's step 1)
