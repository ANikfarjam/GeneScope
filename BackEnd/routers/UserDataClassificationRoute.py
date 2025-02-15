from flask import Blueprint, request, jsonify, send_file
classification_blueprint=Blueprint("classify",__name__)
@classification_blueprint.route("/classify",methods=["GET"])
def get_classify():
    return jsonify("In Progress")