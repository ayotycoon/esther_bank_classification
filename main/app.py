from flasgger import Swagger
from flask import Flask, jsonify
from flask import request

from main.basic_classifier import classifierInstance
from main.bootstrap import Bootstrap

app = Flask(__name__)
swagger = Swagger(app)


@app.route('/', methods=['POST'])
def fff():
    """
    Test a sample transaction
    ---
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: array
          items:
            type: object
            properties:
              title:
                type: string
                example: walmart

    responses:
      200:
        description: Returns a greeting message.
        examples:
          application/json: [{"title":"walmart"}]
    """
    data = request.json
    return jsonify(classifierInstance.get_prediction_as_list(list=data))

with app.app_context():
    Bootstrap.init()

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False,threaded=False)
