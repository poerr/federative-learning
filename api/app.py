from flask import Flask, request, jsonify
from train_nn import train_nn


app = Flask(__name__)


@app.route('/train_nn', methods=['POST'])
def train_nn_endpoint():

    print('ENDPOINT HIT')
    weights = train_nn()

    serialized_weights = [weight.tolist() for weight in weights]    
    return jsonify(serialized_weights), 200


if __name__ == '__main__':
    app.run()

    print(f"Running on port: {app.config['SERVER_NAME'].split(':')[-1]}")