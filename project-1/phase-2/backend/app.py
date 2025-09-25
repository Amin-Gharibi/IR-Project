from flask import Flask, request, jsonify
from flask_cors import CORS
from ir_system import InformationRetrievalSystem

app = Flask(__name__)
CORS(app)

ir_system = InformationRetrievalSystem()


@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize the IR system with dataset and relevance content"""
    try:
        data = request.get_json()
        
        if not data or 'dataset' not in data or 'relevance' not in data:
            return jsonify({
                "status": "error",
                "message": "Missing 'dataset' or 'relevance' field in request body"
            }), 400
        
        dataset_content = data['dataset']
        relevance_content = data['relevance']
        
        result = ir_system.initialize_system(dataset_content, relevance_content)
        
        if result['status'] == 'success':
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500
    

@app.route('/api/search', methods=['POST'])
def search():
    """Search for documents using a query"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({"status": "error", "message": "Missing 'query' field in request body"}), 400
        
        query = data['query']
        top_k = data.get('top_k', 10)
        query_id = data.get('query_id', None)

        if not isinstance(top_k, int) or top_k < 1:
            top_k = 10
        if query_id is not None:
            try:
                query_id = int(query_id)
            except (ValueError, TypeError):
                return jsonify({"status": "error", "message": "'query_id' must be an integer"}), 400
        
        result = ir_system.search(query, top_k, query_id)
        
        if result['status'] in ['success', 'warning']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"status": "error", "message": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"status": "error", "message": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"status": "error", "message": "Internal server error"}), 500


if __name__ == '__main__':
    print("Retrieving the content of default dataset...")
    with open("./CISI.ALL", "r") as file:
        default_dataset_content = file.read()

    with open("./CISI.REL") as file:
        default_dataset_relevance_judgements = file.read()

    with open("./CISI.QRY", "r") as file:
        default_query_content = file.read()

    ir_system.initialize_system(default_dataset_content, default_dataset_relevance_judgements)
    print("IR System initialized successfully.")

    print("System Evaluation process started...")
    result = ir_system.evaluate_system(default_query_content)
    print(f"Evaluation Results:\n{result}")

    print("Starting Information Retrieval API Server...")
    print("API Endpoints available at: http://localhost:8000/api/")
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8000)