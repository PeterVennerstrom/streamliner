import glob
import json

import requests
from hft_src.high_freq_trading_logic import due_diligence

from streamliner.fleet import ModelProxy

paths = glob.glob("./financial_statements/*.jpg")
with open("model_cfg.json", "r") as file:
    model_config = json.load(file)["models"]


def fleet_callable(call_dict):
    url = "http://localhost:5000/fleet_server"

    response = requests.post(url, json=call_dict)
    if response.status_code == 200:
        return response.json()["result"]
    else:
        return response.text


def get_due_diligence(kwargs):
    url = "http://localhost:5000/server_side_due_diligence"

    response = requests.post(url, json=kwargs)
    if response.status_code == 200:
        return response.json()["result"]
    else:
        return response.text


fleet = ModelProxy(model_config, fleet_callable)

print("Run local business logic with calls to model server")
for p in paths:
    rating = due_diligence(fleet, p, extra_due_diligence=True)
    print(p, rating)

print("\n", "Run business logic server side")
for p in paths:
    kwargs = dict(financial_statement=p, extra_due_diligence=True)
    rating = get_due_diligence(kwargs)
    print(p, rating)
