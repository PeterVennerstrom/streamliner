import glob

from hft_src.high_freq_trading_logic import due_diligence

from streamliner.fleet import SingleGPUInfer
from streamliner.model_builder import LocalBuilder

model_builder = LocalBuilder("model_cfg.json", device=0)
fleet = SingleGPUInfer(model_builder)

paths = glob.glob("./financial_statements/*.jpg")

for p in paths:
    rating = due_diligence(fleet, p, extra_due_diligence=True)
    print(p, rating)
