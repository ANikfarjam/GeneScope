from io import StringIO
import pandas as pd
import marimo as mo
import pickle as pkl
import numpy as np
# --- Initial Dataset ---
new_brca_df = pd.read_csv("../AHPresults/initial_clinical_df.csv")

# --- Utility functions ---
def get_value_counts_table(df):
    """Return a long table of (column, value, count) for value_counts across all columns."""
    records = []
    for col in df.columns:
        counts = df[col].value_counts(dropna=False)
        for val, count in counts.items():
            records.append({
                "Column": col,
                "Value": val,
                "Count": count
            })
    return pd.DataFrame(records)

def get_info_str(df):
    """Capture df.info() output as a string."""
    buffer = StringIO()
    df.info(buf=buffer, verbose=True, show_counts=True)
    return buffer.getvalue()

def convert_numeric_columns(df):
    """Convert columns that look numeric into actual numeric types."""
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except Exception:
            pass
    return df
# --- Step 1: Original Dataset ---
original_info = get_info_str(new_brca_df)
original_value_counts = get_value_counts_table(new_brca_df)

# --- Step 2: Replace "NA" with np.nan ---
new_brca_df.replace("NA", np.nan, inplace=True)
after_replace_info = get_info_str(new_brca_df)
after_replace_value_counts = get_value_counts_table(new_brca_df)

# --- Step 3: Convert numeric-looking columns ---
new_brca_df = convert_numeric_columns(new_brca_df)
after_convert_info = get_info_str(new_brca_df)
after_convert_value_counts = get_value_counts_table(new_brca_df)

# --- Step 4: Drop columns after 189th and filter Samples ---
droped_Df = new_brca_df.copy()
droped_Df = droped_Df.iloc[:, :189]
melignent_df = pd.read_csv('../../../data/ModelDataSets/cancerExpressions.csv')
print(melignent_df.head())
droped_Df = droped_Df[droped_Df['Samples'].isin(melignent_df['Unnamed: 0'])]

after_drop_info = get_info_str(droped_Df)
after_drop_value_counts = get_value_counts_table(droped_Df)

# --- Step 5: Create Column Layout Table ---
col_list = droped_Df.columns.tolist()
total_cells = 21 * 9
col_list += [""] * (total_cells - len(col_list))
col_array = np.array(col_list).reshape(21, 9)
col_table_df = pd.DataFrame(col_array, columns=[f"Col {i+1}" for i in range(9)])

# --- Step 6: Final Fill NAs ---
droped_Df.fillna("Not Available", inplace=True)
final_info = get_info_str(droped_Df)
final_value_counts = get_value_counts_table(droped_Df)

ext_code = f"""
# --- Utility functions ---
def get_value_counts_table(df):
    \"\"\"Return a long table of (column, value, count) for value_counts across all columns.\"\"\"
    records = []
    for col in df.columns:
        counts = df[col].value_counts(dropna=False)
        for val, count in counts.items():
            records.append({{
                "Column": col,
                "Value": val,
                "Count": count
            }})
    return pd.DataFrame(records)

def get_info_str(df):
    \"\"\"Capture df.info() output as a string.\"\"\"
    buffer = StringIO()
    df.info(buf=buffer, verbose=True, show_counts=True)
    return buffer.getvalue()

def convert_numeric_columns(df):
    \"\"\"Convert columns that look numeric into actual numeric types.\"\"\"
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except Exception:
            pass
    return df

# --- Step 1: Original Dataset ---
original_info = get_info_str(new_brca_df)
original_value_counts = get_value_counts_table(new_brca_df)

# --- Step 2: Replace "NA" with np.nan ---
new_brca_df.replace("NA", np.nan, inplace=True)
after_replace_info = get_info_str(new_brca_df)
after_replace_value_counts = get_value_counts_table(new_brca_df)

# --- Step 3: Convert numeric-looking columns ---
new_brca_df = convert_numeric_columns(new_brca_df)
after_convert_info = get_info_str(new_brca_df)
after_convert_value_counts = get_value_counts_table(new_brca_df)

# --- Step 4: Drop columns after 189th and filter Samples ---
dropped_Df = new_brca_df.copy()
dropped_Df = dropped_Df.iloc[:, :189]
dropped_Df = dropped_Df[dropped_Df['Samples'].isin(melignent_df['Samples'])]

after_drop_info = get_info_str(dropped_Df)
after_drop_value_counts = get_value_counts_table(dropped_Df)

# --- Step 5: Create Column Layout Table ---
col_list = dropped_Df.columns.tolist()
total_cells = 21 * 9
col_list += [""] * (total_cells - len(col_list))
col_array = np.array(col_list).reshape(21, 9)
col_table_df = pd.DataFrame(col_array, columns=[f"Col {{i+1}}" for i in range(9)])

# --- Step 6: Final Fill NAs ---
dropped_Df.fillna("Not Available", inplace=True)
final_info = get_info_str(dropped_Df)
final_value_counts = get_value_counts_table(dropped_Df)
"""

final_cl1 = mo.ui.tabs({
    'Data': droped_Df.head(100),
    'Info': mo.md(f"```\n{final_info}\n```"),
    'Extraction': mo.ui.code_editor(ext_code)
})

with open('./pkl_files/kl1.pkl', 'wb') as f:
    pkl.dump(final_cl1, f)