
import marimo as mo
import pandas as pd
import pickle as pkl
import io

extraction_code = f"""with open('GSE62944_01_27_15_TCGA_20_420_Clinical_Variables_7706_Samples.txt', 'r') as file:
    exractedData1 = [line.split('\\t') for line in file.readlines()]
with open('GSE62944_06_01_15_TCGA_24_548_Clinical_Variables_9264_Samples.txt', 'r') as file:
    exractedData2 = [line.split('\\t') for line in file.readlines()]

# Convert to DataFrames
df1 = pd.DataFrame(exractedData1)
df2 = pd.DataFrame(exractedData2)

# Transpose the DataFrames
df1 = df1.T
df2 = df2.T

# Concatenate
concat_df = pd.concat([df1, df2])

# Rename columns using the first row
concat_df.columns = concat_df.iloc[0]
concat_df = concat_df.drop(0)  # Remove the first row

# Remove NaN columns
concat_df = concat_df.loc[:, ~concat_df.columns.isna()]

# Rename index column
if '' in concat_df.columns:
    concat_df.rename(columns={{'': 'Samples'}}, inplace=True)

# Capture df.info() output
info_buf = io.StringIO()
concat_df.info(verbose=True, show_counts=True, buf=info_buf)
"""

# Load pre-processed DataFrame for display
concat_df = pd.read_csv('../AHPresults/initial_clinical_df.csv')

# Capture df.info() output separately (you missed this)
info_buf = io.StringIO()
concat_df.info(verbose=True, show_counts=True, buf=info_buf)
info_str = info_buf.getvalue()

# Build Marimo UI
cl_tabs = mo.ui.tabs({
    'Data': mo.ui.table(concat_df.head(100)),
    'Info': mo.md(f"```\n{info_str}\n```"),
    'Extraction': mo.ui.code_editor(extraction_code, language="python")
})

with open("clinical1.plk", 'wb') as f:
    pkl.dump(cl_tabs, f)
print("done!")