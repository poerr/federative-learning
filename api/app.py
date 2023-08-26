from flask import Flask, request, jsonify
from train_nn import train_nn


app = Flask(__name__)


@app.route('/train_nn', methods=['POST'])
def train_nn_endpoint():

    data = request.json
    print(data)
    print(type(data))
    weights = train_nn(data)

    serialized_weights = [weight.tolist() for weight in weights]
    
    return_dictionary = {
        "layer1_weights" : serialized_weights[0],
        "layer1_biases": serialized_weights[1],
        "layer2_weights" : serialized_weights[2],
        "layer2_biases": serialized_weights[3],
        "layer3_weights" : serialized_weights[4],
        "layer3_biases": serialized_weights[5],
        "layer4_weights" : serialized_weights[6],
        "layer4_biases": serialized_weights[7]
    }

    return jsonify(return_dictionary), 200


if __name__ == '__main__':
    app.run()

    print(f"Running on port: {app.config['SERVER_NAME'].split(':')[-1]}")