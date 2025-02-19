from flask import Flask, jsonify
from flask_cors import CORS
from routers.UserDataClassificationRoute import classification_blueprint
from routers.GeneExpressionsranking import GEOAHP_bp
app = Flask(__name__)
# allow universal requests
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})
#register routes
app.register_blueprint(classification_blueprint)
app.register_blueprint(GEOAHP_bp)
 
if __name__ == '__main__':
    app.run(debug=True)