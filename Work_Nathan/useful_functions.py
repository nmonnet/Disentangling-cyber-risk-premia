#-----------------------------------------------------------------------------------------------------------------
#           import relevant libraries, setup options
#-----------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm.notebook import tqdm
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from functools import reduce
from linearmodels.panel import PanelOLS
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import itertools 
from decimal import Decimal
from scipy.stats import percentileofscore
import glob
from sklearn.metrics.pairwise import cosine_similarity
import random
import seaborn as sns
import os
import pickle
import networkx as nx
from scipy.stats import skew
import community
import matplotlib.ticker as ticker
import copy

from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from collections import Counter

from datetime import datetime, timedelta

from scipy import stats
from scipy.stats import norm
from scipy.stats import t
from scipy.stats import ttest_1samp
from scipy.stats import f as f_dist

from itertools import chain, combinations

plt.rcParams["figure.figsize"] = (12,8)

#import functions from functions.py
import sys
sys.path.append('../')  # go up one level to the parent directory
from function_definitions import *

#nltk ressources
import nltk
nltk.download('stopwords')

print("update 22")

#------------------------------------------------------------------------------------------------------------------

#useful dictionnary to seperate cyber risk from data of Daniel Celeny
tactic_dict={
        "Persistence":113,                     #meaning that 0-112 are in "persistence" tactic
        "Command_and_Control":39,   #113-151
        "Impact":26,                                #152-177 [sum of the previous] - [sum-1]
        "Initial_Access":19,                    #178
        "Resource_Development":45,  #197
        "Collection":37,                           #242
        "Exfiltration":18,                          #279
        "Credential_Access":63,           #297
        "Privilege_Escalation":96,         #360
        "Execution":36,                           #456
        "Defense_Evasion":184,           #492
        "Reconnaissance":43,               #676
        "Lateral_Movement":22,            #719
        "Discovery":44                           #741
    }

#another expression to help with separation of cyber risk using index of involved arrays (785)
# Calculate the total number of tactics
total_tactics = sum(tactic_dict.values())

# Create a list to store the names of the tactics
tactics_array = []

# Assign tactic names to the array based on the specified ranges
for tactic, count in tactic_dict.items():
    tactics_array.extend([tactic] * count)

# Convert the list to numpy array if necessary
tactics_array = np.array(tactics_array)

#for pie chart
industries_index = ['Consumer Nondurables', 'Consumer Durables',
                         'Manufacturing', 'Oil and Gas',
                         'Chemicals', 'Business Equipment',
                         'Telephone and Television Transmission',
                         'Utilities', ' Retail', 'Healthcare',
                         'Finance', 'Other']


#-----------------------------------------------------------------------------------------------------------------
#           doc2vec model
#-----------------------------------------------------------------------------------------------------------------

path_best_model_doc2vec='../../models/Doc2vec/model_dbow_v200_e50_w15'
best_model_doc2vec = gensim.models.doc2vec.Doc2Vec.load(path_best_model_doc2vec)

def get_best_model(path_model='../../models/Doc2vec/model_dbow_v200_e50_w15'):
    model = gensim.models.doc2vec.Doc2Vec.load(path_model)

    return model


#-----------------------------------------------------------------------------------------------------------------
#           obtain vector of 10k 
#-----------------------------------------------------------------------------------------------------------------

#path_test_obtain_array_10k='../../data/10k_statements/tokens/2018/AAPL_tokens.csv.gz'

def obtain_array_10k(path):
    df10k=pd.read_csv(path, dtype=str)

    tokens_list = df10k.apply(lambda row: row.dropna().tolist()[1:], axis=1).tolist()

                

    
    vector_list = df10k.apply(lambda row: get_vect(best_model_doc2vec,row.dropna().tolist()[1:],"test"), axis=1)
    vector_array = np.vstack(vector_list.to_numpy())

    return vector_array


#-----------------------------------------------------------------------------------------------------------------
#           obtain vector of MITRES ATTaCK
#-----------------------------------------------------------------------------------------------------------------


def obtain_array_MITRES(path_tactic_='../../data/MITRE_ATT&CK/descriptions/tactics_'):
    #use to obtain the vector representation of all paragraph in MITRES
    #The will be all concatned in a (785,200) array
    #however the array is ordered so you can always retrieve the original paragraph with the index 
    #and taking into account the order in tactic_dict
   
    first=True
    for tactic in list(tactic_dict.keys()):
        path_tactic_spec=path_tactic_+tactic+".csv"
        df_tac=pd.read_csv(path_tactic_spec)
        
        
        
        df_tac["tokens"] = df_tac["Description"].apply(lambda text: get_tokens(text,raw=False,merge=False,sentences=False,sleep=False)[0])
        
        df_tac["vector"]= df_tac["tokens"].apply(lambda tokens: get_vect(best_model_doc2vec, tokens, 'test'))
        
        # Extract the "vector" column from df_tac for the current iteration
        current_vectors =  np.vstack(df_tac["vector"].to_numpy())
    
        # Concatenate the current vectors with the previous ones
        if first:
            concatenated_vectors=current_vectors
            first=False
        else:
            
            concatenated_vectors = np.concatenate((concatenated_vectors, current_vectors), axis=0)


    
    return concatenated_vectors



#-----------------------------------------------------------------------------------------------------------------
#           Compare errors
#-----------------------------------------------------------------------------------------------------------------


def compute_cos_matrix(m1,m2):
    #first 10k then MITRES
    #simple cosinus similarity matrix
    return cosine_similarity(m1, m2)

#path for the cosinus matrix of reference created by Daniel Celeny
pt = '../../data/cyber_risk_measures/doc2vec/per_paragraph/general/2018/AAPL_sim_matrix.csv.gz'

def comparator_error(path_test,cos_matrix_ref):
    #use to compare the error done by using the vector computed with the saved model of D.C.
    #and the saved data by D.C. 

    #path_test contain the saved data
    #cos_matrix_ref contained the matrix constructed with the vectors representation

    
    table1=pd.read_csv(path_test)
    final_score=cos_matrix_ref-table1.values

    flattened_matrix = final_score.flatten()

    # Plot a histogram of the flattened values
    plt.hist(flattened_matrix, bins=100)  # You can adjust the number of bins as needed
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Flattened Matrix Values')
    plt.yscale('log')
    plt.show()


#-----------------------------------------------------------------------------------------------------------------
#           Tactics separator
#-----------------------------------------------------------------------------------------------------------------


def splitter_of_cos_matrix_into_tactics(cos_matrix):
    #split the cos matrix into the different tactic
    dict_cos_matrix={}

    split_down=0
    for key in tactic_dict.keys():
        split_up=split_down+tactic_dict[key]
        
        dict_cos_matrix[key]=cos_matrix[:,split_down:split_up]

        split_down+=tactic_dict[key]

    return dict_cos_matrix


#-----------------------------------------------------------------------------------------------------------------
#           Histogram (density) creation
#-----------------------------------------------------------------------------------------------------------------



def create_histogram(firm):
    #for a given firm, for each year and for each tactic create a dataframe containing an array
    #each array is of the length of the number of paragraph in the 10k 
    #each coordinate of the array correspond to a pargraph highest similitude among the given tactic paragraphs

    #     |tactic1|tactic2|...
    #year1|array11|array12|...
    #year2|array21|array22|...

    #array.shape=(#paragraph in 10K,) 
    #coordinate i contains the highest similitude between all paragraph of MITRES and paragraph i of 10k
    
    df_histo_per_tactic_per_year = pd.DataFrame(columns=list(tactic_dict.keys()))
    for year in np.arange(2008,2024):
       
        
        path_cos_matrix_ref='../../data/cyber_risk_measures/doc2vec/per_paragraph/general/'+str(year)
        path_firm_ref=path_cos_matrix_ref+"/"+firm+"_sim_matrix.csv.gz"

      

        if glob.glob(path_firm_ref):
            
            table=pd.read_csv(path_firm_ref).values
            
            
            dict_cos_matrix=splitter_of_cos_matrix_into_tactics(table)

            for key in dict_cos_matrix.keys():
                #print(key+" : ")
                #print(tactic_dict[key])
                #print(dict_cos_matrix[key].shape)

            
                df_histo_per_tactic_per_year.loc[str(year),key]=np.max(dict_cos_matrix[key],axis=1)
            

        else: 
            print("missing year : ",year)

    #create to observe the distribution of cyber risk across all paragraph of 10k 
    #for each existing tactics

    #find back the overall cyberscore

    df_histo_per_tactic_per_year['Overall cyber score'] = df_histo_per_tactic_per_year.apply(lambda row: np.vstack(row.values), axis=1)

    df_histo_per_tactic_per_year['Overall cyber score'] = df_histo_per_tactic_per_year['Overall cyber score'].apply(lambda x: np.maximum.reduce(x, axis=0))

    #print(histo['Overall cyber score'].iloc[0])
    #print(histo['Overall cyber score'].iloc[0].shape)
    #print(histo['Persistence'].iloc[0].shape)
    #print(histo['Impact'].iloc[0].shape)

    return df_histo_per_tactic_per_year




def create_gif_from_density(histo,firm,output_directory="cyber_score_density_per_tactic/"):
    
 
    for year, row in histo.iterrows():
        palette = sns.color_palette("tab20", 14)
        plt.figure()  # Create a new figure for each year
        for tactic, values in row.items():
            # Calculate skewness, 99th percentile, and 50th percentile
            skewness = skew(values)
            percentile_99 = np.percentile(values, 99)
            percentile_50 = np.percentile(values, 50)

            text_label=tactic.replace("_", " ")
            #text_label+="                                "
            #text_label=text_label[:25]
            #text_label+="50p.="+str(np.round(percentile_50,2))+"  99p.="+str(np.round(percentile_99,2))+" sk.="+str(np.round(skewness,2)) 
            if tactic=='Overall cyber score':
                sns.kdeplot(values, label=text_label,linestyle='--', color='black')
            else:
                sns.kdeplot(values, label=text_label,color=palette.pop(0))  # Plot the density plot
        plt.title(f'Density Plot for year {year} - {firm}')
        plt.xlabel('Cyber score values')
        plt.ylabel('Density of cyber score per tactic')
        plt.xlim([0.1,0.8])
        plt.ylim([0,9])

        plt.legend()

        
        

        

        # Create the directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)

        os.makedirs(output_directory+firm+"/", exist_ok=True)
        
        # Save the plot as an image with the specified filename
        filename = f"{firm}_{year}_cyber_score_per_tactic_density.png"
        plt.savefig(os.path.join(output_directory+firm+"/", filename))
        plt.close()  # Close the plot to release memory
        
    trends=histo.copy()

    def calculate_percentile_99(arr):
        return np.percentile(arr, 99)
    
    # Apply the function to each entry in the dataframe
    percentile_99 = trends.map(calculate_percentile_99)

    trends=histo.copy()

    def calculate_percentile_50(arr):
        return np.percentile(arr, 50)
    
    # Apply the function to each entry in the dataframe
    percentile_50 = trends.map(calculate_percentile_50)


    def calculate_skewness(arr):
        return skew(arr)

    # Apply the function to each entry in the dataframe
    skewness_df = trends.map(calculate_skewness)


    # Plotting
    for column in percentile_99.columns:
        if column=='Overall cyber score':
            plt.plot(percentile_99.index, percentile_99[column], label=column, linestyle='--', color='black')
        else:
            plt.plot(percentile_99.index, percentile_99[column], label=column)
    
    # Adding labels and legend
    plt.xlabel('Year')
    plt.ylabel('99 percentile of cyber score distribution')
    plt.legend()
    plt.title(firm)

    # Save the plot as an image with the specified filename
    filename = f"{firm}_{year}_cyber_score_per_tactic_99p.png"
    plt.savefig(os.path.join(output_directory+firm+"/", filename))
    plt.close()  # Close the plot to release memory

    # Plotting
    for column in percentile_50.columns:
        if column=='Overall cyber score':
            plt.plot(percentile_50.index, percentile_50[column], label=column, linestyle='--', color='black')
        else:
            plt.plot(percentile_50.index, percentile_50[column], label=column)
    
    # Adding labels and legend
    plt.xlabel('Year')
    plt.ylabel('50 percentile of cyber score distribution')
    plt.legend()
    plt.title(firm)

    # Save the plot as an image with the specified filename
    filename = f"{firm}_{year}_cyber_score_per_tactic_50p.png"
    plt.savefig(os.path.join(output_directory+firm+"/", filename))
    plt.close()  # Close the plot to release memory

    # Plotting
    for column in skewness_df.columns:
        if column=='Overall cyber score':
            plt.plot(skewness_df.index, skewness_df[column], label=column, linestyle='--', color='black')
        else:
            plt.plot(skewness_df.index, skewness_df[column], label=column)
    
    # Adding labels and legend
    plt.xlabel('Year')
    plt.ylabel('skewness of cyber score distribution')
    plt.legend()
    plt.title(firm)

    # Save the plot as an image with the specified filename
    filename = f"{firm}_{year}_cyber_score_per_tactic_skew.png"
    plt.savefig(os.path.join(output_directory+firm+"/", filename))
    plt.close()  # Close the plot to release memory



#-----------------------------------------------------------------------------------------------------------------
#           Show and work on similitude matrix
#-----------------------------------------------------------------------------------------------------------------


def setup_similitude_matrix(path_MITRES='../../data/MITRE_ATT&CK/descriptions/tactics_'):
    vectors_MITRES=obtain_array_MITRES(path_MITRES)
    simi_matrix=compute_cos_matrix(vectors_MITRES,vectors_MITRES)

    print("check number of negative similitude : "+str(np.sum(simi_matrix < 0)))
    #we rework a bit the data so that it can be used in clustering methods
    simi_matrix[simi_matrix<0]=0

    
    return simi_matrix


def graph_creator(M):
    #create a spring graph
    m=M.copy()
    np.fill_diagonal(m, 0)
    
    # Create a graph from the similarity matrix
    G = nx.from_numpy_array(m)
    
    # Create a spring layout for the graph
    pos = nx.spring_layout(G)
    
    # Define colors for different tactics
    unique_tactics = np.unique(tactics_array)
    color_map = plt.cm.get_cmap('tab20', len(unique_tactics))  # You can choose a different colormap if you prefer
  
    
    # Assign colors to tactics
    tactic_color_dict = {tactic: color_map(i) for i, tactic in enumerate(unique_tactics)}
    
    # Assign colors to nodes based on their tactic
    node_colors = [tactic_color_dict[tactics_array[i]] for i in range(len(tactics_array))]
    
    # Draw the graph with node colors
    #nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, edge_color='grey', linewidths=1, font_size=7)

    nx.draw_networkx_nodes(G,pos, node_color=node_colors, node_size=500)

    nx.draw_networkx_edges(G,pos,edge_color='grey',alpha=0.02) 


    nx.draw_networkx_labels(G, pos, labels={node: str(node) for node in G.nodes()}, font_size=7)
    
    
    # Create legend
    legend_labels = {tactic: plt.Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=tactic_color_dict[tactic]) for tactic in unique_tactics}
    plt.legend(legend_labels.values(), legend_labels.keys(), loc='best', title='Tactics')
    
    # Show the graph
    #plt.title("Spring Layout Graph from Similarity Matrix")
    plt.show()
    


def force_clustering(similarity_matrix,intensity=1):
    #force paragraph of the same tactic to have a chosen similitude with the intensity parameter
    split_down=0
    first=True
    for key in tactic_dict.keys():
        

        M=np.ones((tactic_dict[key],tactic_dict[key]))
        if first:
            forced_matrix=M

            first=False

        else:
        
            forced_matrix=np.block([ [forced_matrix,np.zeros((forced_matrix.shape[0],M.shape[1]))],
                                [np.zeros((M.shape[0],forced_matrix.shape[1])), M]])


    output=similarity_matrix.copy()

    output=output*(1-forced_matrix)+forced_matrix*intensity

    return output
    

def show_similitude_matrix(simi):
    # Plot the matrix with a colormap
    extent = [0, simi.shape[1],  simi.shape[0],0]
    plt.imshow(simi, cmap='summer', origin='upper', extent=extent) # 'viridis' is just one example of a colormap
    plt.colorbar(orientation="horizontal",fraction=0.046, pad=0.04)  # Add a colorbar to show the mapping of values to colors
    
    
    
    last_split=0
    split_line=0
    for key in tactic_dict.keys():
        split_line+=tactic_dict[key]
    
        plt.text(800, split_line-(split_line-last_split)/2, key.replace("_", " "), color='black', fontsize=8, rotation=0)
    
        plt.text(split_line-(split_line-last_split)/2, -15, key.replace("_", " "), color='black', fontsize=8, rotation=45)
        
        #print(split_line)
        
        if split_line>=784:
            break
        plt.hlines(split_line-1.5,0,784,color="black",linewidths=0.5)
        plt.vlines(split_line,0,784,color="black",linewidths=0.5)
    
        last_split=split_line

        
        
    plt.show()


#-----------------------------------------------------------------------------------------------------------------
#           Clustering
#-----------------------------------------------------------------------------------------------------------------

#important : check that the labels is in a good format:
#array size 785
#contain class ranging from 0 to n-1

def comparator_tactic_super_tactic(labels,n_cluster=None,show_plot=True):
    if n_cluster==None:
        n_cluster=np.max(labels)+1


    tactics_name=list(tactic_dict.keys())
    overall_occurrences = np.array([np.count_nonzero(tactics_array == value) for value in tactics_name])

    entropy_shannon=0

    first=True
    
    for k in range(n_cluster):
        classe_occurences=np.array([np.count_nonzero(tactics_array[labels==k] == value) for value in tactics_name])

        proportion_in_classe=classe_occurences/overall_occurrences

        if first:
            array_proportion_in_classe=proportion_in_classe
            first=False

        else:
            array_proportion_in_classe=np.vstack((array_proportion_in_classe,proportion_in_classe))

        entropy_shannon+=-np.where(proportion_in_classe != 0, proportion_in_classe * np.log(proportion_in_classe), 0)

    #now we built the indicator of color
    
    colors_indicator=np.argmax(array_proportion_in_classe, axis=0)


    if show_plot:

        for k in range(n_cluster):    
            

            colors = ['red' if ind == k else 'skyblue' for ind in colors_indicator]
           
            plt.bar(tactics_name, array_proportion_in_classe[k], color=colors)
            plt.xlabel('Tactics')
            plt.ylabel('Proportion of tactics in class '+str(k))
            plt.ylim([0,1])
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

          
        
        
        plt.bar(np.array(tactics_name),entropy_shannon, color="green")
        plt.xlabel('Tactics')
        plt.ylabel('Estimated Shannon entropy for each tactic after the clustering ')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    print("lower entropy means higher exclusivity to a 'super tactic'")
    print("sum of all exclusive entropy = ",np.sum(entropy_shannon))

    return np.sum(entropy_shannon)






#-----------------------------------------------------------------------------------------------------------------
#           Clustering presentation results
#-----------------------------------------------------------------------------------------------------------------

def organiser1(lab):
    to_switch_with_0 = np.argmax(np.bincount(lab[197:241]))
    

    transformed_lab = [0 if num == to_switch_with_0 else to_switch_with_0 if num == 0 else num for num in lab]
    return np.array(transformed_lab)


def organiser2(lab):
    to_switch_with_1 = np.argmax(np.bincount(lab[456:491]))
    

    transformed_lab = [1 if num == to_switch_with_1 else to_switch_with_1 if num == 1 else num for num in lab]
    return np.array(transformed_lab)



def organiser3(lab):
    n_cluster=np.max(lab)+1
    if n_cluster>2:
        to_switch_with_2 = np.argmax(np.bincount(lab[719:740]))
        
    
        transformed_lab = [2 if num == to_switch_with_2 else to_switch_with_2 if num == 2 else num for num in lab]
        return np.array(transformed_lab)
    else:
        return lab



def comparator_method_clustering(dict,fig_size=(20, 30)):
    #number_rows
    number_rows=0
    for lab in dict.values():
        number_rows=np.max(np.array([np.max(lab),number_rows]))

    number_rows+=2

    #number_columns
    number_columns=len(dict.keys())

    # Create a figure and axis objects
    fig, axs = plt.subplots( number_columns,number_rows, figsize=fig_size)



    #useful for the second graph
    method_entropy=[]
    method_balanced=[]

    
    for i,method in enumerate(dict.keys()): #over columns
        
            labels=dict[method]
            
            labels=organiser3(labels) #don't change the order please!
            labels=organiser1(labels)
            labels=organiser2(labels)
            
            #the organiser allows to have roughly corresponding name of class across the method
            

           
            n_cluster=np.max(labels)+1
        
        
            tactics_name=list(tactic_dict.keys())
            overall_occurrences = np.array([np.count_nonzero(tactics_array == value) for value in tactics_name])
        
            entropy_shannon=0
        
            first=True
            
            for k in range(n_cluster):
                classe_occurences=np.array([np.count_nonzero(tactics_array[labels==k] == value) for value in tactics_name])
        
                proportion_in_classe=classe_occurences/overall_occurrences
        
                if first:
                    array_proportion_in_classe=proportion_in_classe
                    first=False
        
                else:
                    array_proportion_in_classe=np.vstack((array_proportion_in_classe,proportion_in_classe))
        
                entropy_shannon+=-np.where(proportion_in_classe != 0, proportion_in_classe * np.log(proportion_in_classe), 0)
        
            #now we built the indicator of color
            
            colors_indicator=np.argmax(array_proportion_in_classe, axis=0)
        
        
            
        
            for k in range(n_cluster):    
                
    
                colors = ['red' if ind == k else 'skyblue' for ind in colors_indicator]


                axs[i,k].bar(np.arange(14)+1, array_proportion_in_classe[k], color=colors)
                axs[i,k].xaxis.set_major_locator(ticker.MultipleLocator(base=2))
                axs[i,k].set_ylim([0,1])
                
                if k==0:
                    #axs[i,k].set_title('0')
                    axs[i,k].set_ylabel(method+'\n\n'+'Prop. class '+str(k), fontsize=12)

                else:
                    axs[i,k].set_ylabel('Prop. class '+str(k), fontsize=12)


        
           
            if n_cluster<number_rows-1:
                axs[i,number_rows-2].bar([1,5,14], [0,1,0.5], color="white")
                axs[i,number_rows-2].set_ylim([0,1])

            #compute balanced_score
            balanced_score=np.round(np.std(np.array(list(Counter(labels).values()))),1)

            


        
            text='entropy sum. = '+str(np.round(np.sum(entropy_shannon),2))+'\n\n'+'balanced score = '+str(np.round(balanced_score,1))
            axs[i,number_rows-2].text(2, 0.5, text, fontsize=12, color='black')  

            method_entropy.append(np.sum(entropy_shannon))
            method_balanced.append(balanced_score)
            

            
            axs[i,number_rows-1].bar(np.arange(14)+1,entropy_shannon, color="green")
            axs[i,number_rows-1].set_ylabel('Entropy', fontsize=12)
            axs[i,number_rows-1].xaxis.set_major_locator(ticker.MultipleLocator(base=2))
            axs[i,number_rows-1].set_ylim([0,1.5])

            for k in range(n_cluster,number_rows-1):   
                axs[i,k].set_xticks([])
                axs[i,k].set_yticks([])
                
            
               


    plt.tight_layout()
    plt.show()

    
    plt.scatter(method_balanced,method_entropy)
    for i,method in enumerate(dict.keys()):
        plt.text(method_balanced[i],method_entropy[i]+0.25,method,fontsize=6,ha="center")

    plt.xlabel("Score of balance between classes")
    plt.ylabel("Entropy (heterogeneity between classes)")
    plt.show()




def graph_creator_with_labels(M,lab):
    #create a spring graph
    m=M.copy()
    np.fill_diagonal(m, 0)
    
    # Create a graph from the similarity matrix
    G = nx.from_numpy_array(m)
    
    # Create a spring layout for the graph
    pos = nx.spring_layout(G)
    
    # Define colors for different tactics
    unique_tactics = np.unique(tactics_array)
    color_map = plt.cm.get_cmap('tab20', len(unique_tactics))  # You can choose a different colormap if you prefer
  
    
    # Assign colors to tactics
    tactic_color_dict = {tactic: color_map(i) for i, tactic in enumerate(unique_tactics)}
    
    # Assign colors to nodes based on their tactic
    node_colors = [tactic_color_dict[tactics_array[i]] for i in range(len(tactics_array))]
    
    # Draw the graph with node colors
    #nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, edge_color='grey', linewidths=1, font_size=7)

    # Define colors for different classes
    class_colors = {0: 'm', 1: 'k', 2: 'b', 3: 'y', 4: 'c', 5: 'r'}  # Add colors for labels 4, 5, and 6

    # Generate node colors based on class labels
    outline_colors = [class_colors[label] for label in lab]

    #nx.draw_networkx_nodes(G,pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, edgecolors=outline_colors, linewidths=2)

    nx.draw_networkx_edges(G,pos,edge_color='grey',alpha=0.02) 


    nx.draw_networkx_labels(G, pos, labels={node: str(node) for node in G.nodes()}, font_size=7)
    
    
    # Create legend
    legend_labels = {tactic: plt.Line2D([0], [0], marker='o', color='w', markersize=8, markerfacecolor=tactic_color_dict[tactic]) for tactic in unique_tactics}
    plt.legend(legend_labels.values(), legend_labels.keys(), loc='best', title='Tactics')
    
    # Show the graph
    #plt.title("Spring Layout Graph from Similarity Matrix")
    plt.show()



#-----------------------------------------------------------------------------------------------------------------
#           data extractor 
#-----------------------------------------------------------------------------------------------------------------



def subtract_one_month(date_str):
    # Convert string to datetime object
    original_date = datetime.strptime(date_str, '%Y-%m-%d')
    
    # Subtract one month
    result_date = original_date - timedelta(days=31)
    
    # Format result_date back to string
    result_date_str = result_date.strftime('%Y-%m-%d')
    
    return result_date_str


#I add to redo this function from DC because it was not taking into account "None"
def match_FF_industry(SIC, SIC_to_FF):
    if SIC==None:
        return np.nan
    
    if np.isnan(SIC):
        return np.nan
    try:
        return SIC_to_FF[(SIC_to_FF.SIC_low <= SIC) & (SIC_to_FF.SIC_high >= SIC)].FF_12.values[0]
    except:
        #others = 12
        return 12



#-----------------------------------------------------------------------------------------------------------------
#           transaction cost / bid-ask estimator 
#-----------------------------------------------------------------------------------------------------------------

def de_meaned(r,tau):
    return r-tau*np.mean(r)/np.mean(tau)




#work with following line:
#price_table["bid ask factor"]= price_table[['prc']].rolling(window='31D').apply(bid_ask_estimator, args=(price_table,)).values


def bid_ask_estimator(series, prc_tab):
    #ardiaGuidottiKroencke2023W
    
    last_prices = prc_tab.loc[series.index]
    
    cl_1=np.log(last_prices['prc'].values)[:-1]     #c{t-1}
    op=np.log(last_prices['openprc'].values)[1:]  #o{t}
    lo=np.log(last_prices['bidlo'].values)[1:]    #l{t}
    hi=np.log(last_prices['askhi'].values)[1:]    #h{t}

    hi_1=np.log(last_prices['askhi'].values)[:-1]
    lo_1=np.log(last_prices['bidlo'].values)[:-1]
    
 
    if hi.shape[0]/31<0.5: #less than 50% of the prices are available in the range, we cannot make a decent estimation
        return np.nan

    else:
        
        eta=(lo+hi)/2 #eta{t}
        eta_1=(lo_1+hi_1)
        tau = np.where((cl_1 == hi) & (cl_1 == lo), 0, 1) #tau{t}

        
        #pi_0 :
        p1=np.mean(np.where((op != hi) & (tau == 1), 1, 0))
        p2=np.mean(np.where((op != lo) & (tau == 1), 1, 0))

        pi_0=-8/(p1+p2)

        #pi_c :
        p1=np.mean(np.where((cl_1 != hi_1) & (tau == 1), 1, 0))
        p2=np.mean(np.where((cl_1 != lo_1) & (tau == 1), 1, 0))

        pi_c=-8/(p1+p2)



        
        x1=pi_0/2*de_meaned(eta-op,tau)*(op-eta_1)+pi_c/2*de_meaned(eta-cl_1,tau)*(cl_1-eta_1)
        
        x2=pi_0/2*de_meaned(eta-op,tau)*(op-cl_1)+pi_c/2*de_meaned(op-cl_1,tau)*(cl_1-eta_1)


        w1=np.var(x2)/(np.var(x1)+np.var(x2))
        w2=1-w1

        S2=w1*np.mean(x1)+w2*np.mean(x2)


        #must discussed which one we use

        S=np.sqrt(np.maximum(S2,0)) 

        #S=np.sqrt(abs(S2))
        #negative value are certainly due to negative fluctuation and tend to be bigger
        #we don't want to inflate our transaction cost measure so we drop this alternative
        
        

        
        return S


#-----------------------------------------------------------------------------------------------------------------
#           text variables  
#-----------------------------------------------------------------------------------------------------------------

#alternative version of DC functions "get_tokens()"
def get_cleaned_text(text, ticker, raw = True, sentences = True, find_item1A_ = False):
    #if the input is raw, extract the text
    if raw:
        if find_item1A_:
            document, item1A = get_text(text, find_item1A_ = True)
            if item1A != 'ITEM 1A NOT FOUND':
                #delete extra spaces and short sentences
                item1A = re.sub('\n',' ', item1A)
                item1A = re.sub(r'&nbsp', ' ', item1A)
                item1A = re.sub(r'\xa0;', ' ', item1A)
                item1A = re.sub(r'&#\d+;', ' ', item1A)
                item1A = re.sub('<.*?>', ' ', item1A)
                item1A = re.sub('\s{2,}',' ', item1A)
                lines_1A = sent_tokenize(item1A)
                lines_1A = [line for line in lines_1A if len(word_tokenize(line)) > 15]
        else:
            document = get_text(text)
        #delete extra spaces and short sentences
        document = re.sub('\n',' ', document)
        document = re.sub(r'&nbsp', ' ', document)
        document = re.sub(r'\xa0;', ' ', document)
        document = re.sub(r'&#\d+;', ' ', document)
        document = re.sub('<.*?>', ' ', document)
        document = re.sub('\s{2,}',' ', document)
        lines = sent_tokenize(document)
        lines = [line for line in lines if len(word_tokenize(line)) > 15]
        
        if find_item1A_:
            if item1A != 'ITEM 1A NOT FOUND':
                try:
                    indices_lines = [i for i, x in enumerate(lines) if x in lines_1A]
                    indices_lines_1A = [lines_1A.index(x) for x in lines if x in lines_1A]
                    
                    first_1A = indices_lines[np.argmin(indices_lines_1A)]
                    indices_lines_1A = indices_lines_1A[np.argmin(indices_lines_1A):]
                    last_1A = indices_lines[np.argmax(indices_lines_1A)]
                    
                    if last_1A < first_1A:
                        first_1A = np.nan
                        last_1A = np.nan
                except:
                    first_1A = np.nan
                    last_1A = np.nan
            else:
                first_1A = np.nan
                last_1A = np.nan
    else:
        if sentences:
            lines = sent_tokenize(text)
        else:
            lines = [text]


    if find_item1A_:
        return (ticker,lines,first_1A,last_1A+1,len(text)) 

    else:
        return (ticker,lines,np.nan,np.nan,len(text))


#obtain the variable "secrets" as defined in florackis
def secrets(list_text):
    for j,_ in enumerate(list_text):
        if j>0 and j<len(list_text)-1:
            text=list_text[j-1]+" "+list_text[j]+" "+list_text[j+1]

            
            key_phrases = ["trade secret", "trade secrets", "confidential information", "proprietary information","confidential informations", "proprietary informations","proprietaries informations""proprietaries information"]
            protection_terms = ["protect", "protection", "safeguard"]
        
            # Convert text to lowercase for case insensitivity
            text_lower = text.lower()
        
            # Search for key phrases and protection terms
            for phrase in key_phrases:
                # Look for the key phrase with optional capitalization
                pattern = re.compile(r'\b{}\b'.format(phrase), re.IGNORECASE)
                match = pattern.search(text_lower)
                if match:
                    # Within a 5-word window, check for protection terms
                    window_start = max(0, match.start() - 15)  # 5 words * 3 characters per word
                    window_end = min(len(text), match.end() + 15)
                    window_text = text_lower[window_start:window_end]
                    for term in protection_terms:
                        if term in window_text:
                            #return (1,j)
                            return 1
        
    #return (0,np.nan)
    return 0


#obtain the variable "Risk section length" as defined in florackis
def risk_section_length(first_ind, last_ind):
    if np.isnan(first_ind) or np.isnan(last_ind):
        return 0
    else:
        return last_ind - first_ind









#-----------------------------------------------------------------------------------------------------------------
#           cyberscore, construction of dataframe 
#-----------------------------------------------------------------------------------------------------------------

def select_cyber_score_from_histo(histo_array):
    # Calculate the top 1% value using quantile
    top_value = np.quantile(histo_array, 0.99)
    
    # Select values greater than or equal to the top 1% value
    top_values = histo_array[histo_array >= top_value]
    
    # Calculate the average of the top 1% values
    avg_top_values = np.mean(top_values)

    return avg_top_values
            



def organize_in_super_tactic(dict):
    #the group were formed using the louvain method on the similitude matrix with high and low treshold

    output={}
    
    #group 1 
    name1="Preparation and Reconaissance"
    list1=["Impact", "Initial_Access", "Resource_Development", "Reconnaissance", "Discovery"]

    m1=dict[list1[0]]
    for i,key in enumerate(list1):
        if i>0:
            m1=np.concatenate((m1, dict[key]), axis=1)
    
    #group 2
    name2="Persistence and Evasion"
    list2=["Persistence", "Privilege_Escalation", "Execution", "Defense_Evasion"]

    m2=dict[list2[0]]
    for i,key in enumerate(list2):
        if i>0:
            m2=np.concatenate((m2, dict[key]), axis=1)

    #group 3
    name3="Credential Movement"
    list3=["Credential_Access","Lateral_Movement"]

    m3=dict[list3[0]]
    for i,key in enumerate(list3):
        if i>0:
            m3=np.concatenate((m3, dict[key]), axis=1)

    #group 4
    name4="Command and Data Manipulation"
    list4=["Command_and_Control","Collection","Exfiltration"]

    m4=dict[list4[0]]
    for i,key in enumerate(list4):
        if i>0:
            m4=np.concatenate((m4, dict[key]), axis=1)


    output[name1]=m1
    output[name2]=m2
    output[name3]=m3
    output[name4]=m4

    #print("sanity check : ")
    #print(m1.shape[1]+m2.shape[1]+m3.shape[1]+m4.shape[1])

    return output
    

#-----------------------------------------------------------------------------------------------------------------
#           Fama French factors, momentum, 12 industries, market, rf
#-----------------------------------------------------------------------------------------------------------------
def get_industry_12_portfolio():
    industry_12_portfolio = pd.read_csv('additional_Fama_French_factors/12_Industry_Portfolios.csv', skiprows=11,nrows=1172)
    industry_12_portfolio.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    array_date=[]
    for date_num in industry_12_portfolio["date"]:
        date_string=str(date_num)
        transformed_date = date_string[:4] + '-' + date_string[4:6]
        array_date.append(transformed_date)
    
    array_date=np.array(array_date)
    industry_12_portfolio["date"]=array_date
    industry_12_portfolio.set_index('date', inplace=True)

    return industry_12_portfolio/100


def get_FF_5_Momentum():
    FF_5_Momentum=pd.read_csv('additional_Fama_French_factors/F-F_Momentum_Factor.csv', skiprows=12,nrows=1166)
    FF_5_Momentum.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    array_date=[]
    for date_num in FF_5_Momentum["date"]:
        date_string=str(date_num)
        transformed_date = date_string[:4] + '-' + date_string[4:6]
        array_date.append(transformed_date)
    
    array_date=np.array(array_date)
    FF_5_Momentum["date"]=array_date
    FF_5_Momentum.set_index('date', inplace=True)

    
    # Rename the column to 'New_Column'
    FF_5_Momentum = FF_5_Momentum.rename(columns={FF_5_Momentum.columns[0]: 'UMD'})

    return FF_5_Momentum/100


def get_FF_5_factors():
    FF_5_factors=pd.read_csv('additional_Fama_French_factors/F-F_Research_Data_5_Factors_2x3.csv', skiprows=3,nrows=728)
    FF_5_factors.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    array_date=[]
    for date_num in FF_5_factors["date"]:
        date_string=str(date_num)
        transformed_date = date_string[:4] + '-' + date_string[4:6]
        array_date.append(transformed_date)
    
    array_date=np.array(array_date)
    FF_5_factors["date"]=array_date
    FF_5_factors.set_index('date', inplace=True)

    return FF_5_factors/100




def get_FF_5_daily_factors():
    FF_5_factors=pd.read_csv('additional_Fama_French_factors/F-F_Research_Data_5_Factors_2x3_daily.csv', skiprows=3)
    FF_5_factors.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    array_date=[]
    for date_num in FF_5_factors["date"]:
        date_string=str(date_num)
        transformed_date = date_string[:4] + '-' + date_string[4:6]+ '-' + date_string[6:]
        array_date.append(transformed_date)
    
    array_date=np.array(array_date)
    FF_5_factors["date"]=array_date
    FF_5_factors.set_index('date', inplace=True)

    return FF_5_factors/100



