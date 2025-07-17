import json
import random

with open("params.json", "w") as f:
    json.dump({"bias": 0.0, "weight": random.uniform(-0.01, 0.01)}, f)