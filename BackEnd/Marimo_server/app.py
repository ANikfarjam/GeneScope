import marimo

__generated_with = "0.12.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    import base64
    def create_card(icon_html: str, title: str, desc: str, button_label: str):
        return mo.md(f"""
        <style>
        .card-container {{
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 20px;
            padding: 20px;
        }}
        .card {{
            width: 250px;
            height: 300px;
            border: 1px solid #ccc;
            padding: 16px;
            border-radius: 8px;
            background: #fff;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        .card-icon img {{
            display: block;
            margin: 0 auto 10px;
            width: 50px;
            height: 50px;
        }}
        .card-title {{
            font-size: 1.2em;
            margin-bottom: 10px;
        }}
        .card-desc {{
            flex-grow: 1;
            margin-bottom: 10px;
        }}
        .card-button {{
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            background-color: #007bff;
            color: white;
            cursor: pointer;
        }}
        </style>
        <div class="card">
            <div class="card-icon">{icon_html}</div>
            <div class="card-title"><strong>{title}</strong></div>
            <div class="card-desc">{desc}</div>
            <button class="card-button">{button_label}</button>
        </div>
        """)

    # Define the icon HTML (e.g., an image tag)
    with open("chip.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    data_uri = f"data:image/png;base64,{encoded_string}"

    icon_ml = f'<img src="{data_uri}" alt="ML Icon">'

    # Create cards
    card1 = create_card(
        icon_html=icon_ml,
        title="Machine Learning",
        desc="Explore predictive models used for breast cancer classification and compare their performance.",
        button_label="Learn More"
    )
    # Define the icon HTML (e.g., an image tag)
    with open("exploratory-analysis.png", "rb") as image_file:
        encoded_string2 = base64.b64encode(image_file.read()).decode('utf-8')
    data_uri2 = f"data:image/png;base64,{encoded_string2}"
    icon_ml2 = f'<img src="{data_uri2}" alt="EDA Icon">'
    card2 = create_card(
        icon_html=icon_ml2,
        title="Objective and Data Exploration",
        desc="Explore extracted data, perform feature selection, and prepare data from supplementary files.",
        button_label="Read More"
    )
    # Define the icon HTML (e.g., an image tag)
    with open("monitor.png", "rb") as image_file:
        encoded_string3 = base64.b64encode(image_file.read()).decode('utf-8')
    data_uri3 = f"data:image/png;base64,{encoded_string3}"
    icon_ml3 = f'<img src="{data_uri3}" alt="EDA Icon">'
    card3 = create_card(
        icon_html=icon_ml3,
        title="Major Findings",
        desc="Perform deep analysis of breast cancer biomarkers and prognosis with intuitive visuals.",
        button_label="Read More"
    )

    # Display the cards within a container
    mo.md(f"""
    <div class="card-container">
        {card1}
        {card2}
        {card3}
    </div>
    """)
    return (
        base64,
        card1,
        card2,
        card3,
        create_card,
        data_uri,
        data_uri2,
        data_uri3,
        encoded_string,
        encoded_string2,
        encoded_string3,
        icon_ml,
        icon_ml2,
        icon_ml3,
        image_file,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""We implemented this data exploration tools for user to explore our data and provide ability to gain some insight vrom the visuals:""")
    return


@app.cell(hide_code=True)
def _(mo):
    import pandas as pd
    # Load datasets
    ahp_df = pd.read_csv('./AHPresults/final_Mod_ahp_scores.csv')
    stage_df = pd.read_csv('./AHPresults/fina_Stage_unaugmented.csv')
    healthy_exp_df = pd.read_csv('../../data/ModelDataSets/helthyExpressions.csv')
    cancer_exp_df = pd.read_csv('../../data/ModelDataSets/cancerExpressions.csv')

    # Dictionary of datasets
    datasets = {
        "AHP Analysis": ahp_df,
        "Cancer Stage DataSet": stage_df,
        "Healthy Mean Expression": healthy_exp_df,
        "Cancer Mean Expression": cancer_exp_df,
    }

    # UI: Dropdown
    dropdown = mo.ui.dropdown(
        label="Select Dataset",
        options=list(datasets.keys()),
        value="AHP Analysis"  # default
    )
    return (
        ahp_df,
        cancer_exp_df,
        datasets,
        dropdown,
        healthy_exp_df,
        pd,
        stage_df,
    )


@app.cell(hide_code=True)
def _(datasets, dropdown, mo):
    def show_data():
            selected = dropdown.value
            df = datasets[selected]
            return mo.ui.data_explorer(df)
    mo.vstack([
        dropdown,
        show_data()
    ])
    return (show_data,)


if __name__ == "__main__":
    app.run()
