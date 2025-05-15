from rich.progress import Progress
import firebase_admin
from firebase_admin import credentials, db, firestore
import os
from os import path
from dotenv import load_dotenv
import pandas as pd
import pickle
load_dotenv()
# Initialize Firebase
cred = credentials.Certificate("./FB/genescope-c9328-firebase-adminsdk-fbsvc-9cffaec2d5.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': os.getenv('databaseURL')
})

# Initialize Firestore (after app init)
firestore_db = firestore.client()

"""
1- Fetch AHP scores
2- pick up top 500 
3- send them to FB
"""
print("Loading AHP resul dataset and related vars!")
ahp_df = pd.read_csv('./AHPresults/final_Mod_ahp_scores.csv')
gene_names = ahp_df['Gene'].to_list()
ahp_df = ahp_df.sort_values(by='Scores')
top_genes = ahp_df['Gene'].to_list()[:500]

#select the top 1000 genes
def get_matrix_df(gene_names, top_genes,sparse_matrix):
    converted_matrix = sparse_matrix.toarray()
    result_df = pd.DataFrame(converted_matrix)
    result_df.index = gene_names
    result_df.columns = gene_names
    result_df.fillna(0, inplace=True)
    result_df = result_df.loc[top_genes,top_genes]
    return result_df


# def process_pwm(pwm, pwm_name):
#     ref = db.reference(f"pairwise_matrices/{pwm_name}")
#     pwm_dict = pwm.to_dict()
#     for row_key, row_val in pwm_dict.items():
#         ref.child(row_key).set(row_val)
#     print(f'{pwm_name} pushed to FB successfully')

def sanitize_key(key):
    # Replace illegal characters with underscores
    return str(key).replace('.', '_').replace('$', '_').replace('#', '_')\
                   .replace('[', '_').replace(']', '_').replace('/', '_')

def process_pwm(pwm, pwm_name):
    ref = db.reference(f"pairwise_matrices/{pwm_name}")
    pwm_dict = pwm.to_dict()
    
    for row_key, row_val in pwm_dict.items():
        safe_row_key = sanitize_key(row_key)
        safe_row_val = {sanitize_key(col): val for col, val in row_val.items()}
        ref.child(safe_row_key).set(safe_row_val)
    
    print(f'{pwm_name} pushed to FB successfully')
def process_pwm_firestore(pwm_df, pwm_name):
    collection_ref = firestore_db.collection('PWMs')
    doc_ref = collection_ref.document(pwm_name)
    
    pwm_dict = pwm_df.to_dict(orient='index')  # rows as keys
    
    sanitized_dict = {}
    for row_key, row_val in pwm_dict.items():
        safe_row_key = sanitize_key(row_key)
        safe_row_val = {sanitize_key(col): float(val) for col, val in row_val.items()}
        sanitized_dict[safe_row_key] = safe_row_val

    doc_ref.set(sanitized_dict)
    print(f'{pwm_name} pushed to Firestore successfully')
if __name__=='__main__':
    with Progress() as progress:
        task1= progress.add_task(f'Loading PWMs', total=4)
        #load the pwm
        print("Loading PWMs!")
        with open('./AHPresults/t_test_pwm.pkl', 'rb') as file:
            t_test_pwm = pickle.load(file) 
        progress.update(task1, advance=1)
        with open('./AHPresults/entropy_pwm.pkl', 'rb') as file:
            entropy_pwm=pickle.load(file)
        progress.update(task1, advance=1)
        with open('./AHPresults/roc_auc_pwm.pkl', 'rb') as file:
            roc_pwm = pickle.load(file)
        progress.update(task1, advance=1)
        with open('./AHPresults/snr_pwm.pkl', 'rb') as file:
            snr_pwm = pickle.load(file)
        progress.update(task1, advance=1)
        print("PWMs imported!")

        #create PWM dataframe labling the columns and rows and extract the top 500
        task2=progress.add_task('creating top 500 pwm data frame', total=4)
        #process ttest pwm
        print('Processing t_test pwm!')
        t_test_pwm_mtrx = get_matrix_df(gene_names=gene_names, top_genes=top_genes, sparse_matrix=t_test_pwm)
        process_pwm(t_test_pwm_mtrx ,'t_test')
        progress.update(task2, advance=1)
        #process entropy pwm
        print('Processing entropy pwm!')
        entropy_pwm_mtrx = get_matrix_df(gene_names=gene_names, top_genes=top_genes, sparse_matrix=entropy_pwm)
        process_pwm(entropy_pwm_mtrx, 'entropy')
        progress.update(task2, advance=1)
        #process roc pwm
        print('Processing roc pwm!')
        roc_pwm_mtrx = get_matrix_df(gene_names=gene_names, top_genes=top_genes, sparse_matrix=roc_pwm)
        process_pwm(roc_pwm_mtrx, 'roc')
        progress.update(task2, advance=1)
        #process snr pwm
        print('Processing snr pwm!')
        snr_pwm_mtrx = get_matrix_df(gene_names=gene_names, top_genes=top_genes, sparse_matrix=snr_pwm)
        process_pwm(snr_pwm_mtrx, 'snr')
        progress.update(task2, advance=1)
        print('Task completed!')
        
        

