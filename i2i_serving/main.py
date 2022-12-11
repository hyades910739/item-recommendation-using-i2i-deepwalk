from flask import Flask, request, Response
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List
import json

app = Flask(__name__)

FILENAME = os.environ.get('I2I_FILENAME', 'i2i_map.json')

def _load_i2i_map_file() -> Dict[str, List[str]]:
    base_path = Path(__file__).parent
    filename = base_path / FILENAME
    with open(filename, 'rt') as f:
        dic = json.load(f)
    app.logger.info('load i2i map successfully.')
    return dic

I2I_MAP = _load_i2i_map_file()


@app.route('/predict', methods=['POST'])
def predict():
    req = json.loads(request.json)
    k = req.get('k')
    item = req.get('item')
    response = json.dumps(dict(
        items=I2I_MAP[item][:k]
    ))
    return Response(response, status=200)


if __name__ == "__main__":
    arg_parser = ArgumentParser(
        usage='Usage: python ' + __file__ + ' [--port <port>] [--help]'
    )
    arg_parser.add_argument('-p', '--port', default=9000, help='port')
    arg_parser.add_argument('-d', '--debug', action='store_true', default=False, help='debug')
    options = arg_parser.parse_args()

    app.run(host='0.0.0.0', debug=options.debug, port=options.port)


