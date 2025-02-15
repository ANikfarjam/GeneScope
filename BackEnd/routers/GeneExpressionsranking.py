from flask import Blueprint, request, jsonify, send_file
import pandas as pd
GEOAHP_bp=Blueprint("GEOAHPRanking_Extraction",__name__)
@GEOAHP_bp.route("/get_geoAHPranked", methods=["GET"])
def get_AHP():
    data=pd.read_csv("../../data/AHPRankedGenes.csv")
    data_json=data.to_json()
    return data_json