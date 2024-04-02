import multiprocessing as mp

from flask import Flask, jsonify, request
from hft_src.high_freq_trading_logic import due_diligence

from streamliner.fleet import MultiDeviceFleet

app = Flask(__name__)


@app.route("/fleet_server", methods=["POST"])
def process_request():
    call_dict = request.get_json()
    fleet_callable = app.config["fleet_callable"]
    result = fleet_callable(call_dict)

    return jsonify({"result": result})


@app.route("/server_side_due_diligence", methods=["POST"])
def run_due_diligence():
    kwargs = request.get_json()
    fleet = app.config["fleet"]
    result = due_diligence(fleet, **kwargs)

    return jsonify({"result": result})


if __name__ == "__main__":
    mp.set_start_method("spawn")
    model_builder_cfg = {
        "class": "LocalBuilder",
        "init_params": {"path_to_cfg": "model_cfg.json"},
    }

    device_indices = [0, 1]
    md_fleet = MultiDeviceFleet(device_indices, model_builder_cfg)

    app.config["fleet_callable"] = md_fleet.fleet_callable
    app.config["fleet"] = md_fleet.model_proxy

    app.run(debug=True)
