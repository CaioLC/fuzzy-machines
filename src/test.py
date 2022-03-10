# %%
from fuzz.kernel import Kernel
from fuzz.memb_funcs import Linear, Constant, Trimf, Trapmf
from fuzz.rules import AND, OR, IS, NOT
from fuzz.engine import Engine


food = (
    Kernel(0,10)
        .add_memb_func('rancid', Trapmf(-1, 0, 1, 4))
        .add_memb_func('average', Trimf(1,4,8))
        .add_memb_func('good', Trapmf(4, 8, 10, 11))
)

service = (
    Kernel(0,10)
        .add_memb_func('great', Linear(0, 10))
        .add_memb_func('poor', Linear(10, 0))
)

tips = (
    Kernel(10,30)
        .add_memb_func('low', Constant(10,15))
        .add_memb_func('average', Constant(15,25))
        .add_memb_func('high', Constant(25,30))
)

# %%
eng = (
    Engine()
        .add_kernel("food", food)
        .add_kernel("service", service)
        .add_inference_kernel(tips)
)
eng.add_rule('low', OR(IS({'food':'rancid'}), AND({'food': 'average'}, {'service': 'poor'})))
eng.add_rule('average', AND({"food": "good"}, {"service": "poor"}))
eng.add_rule('average', AND({"food": "average"}, {"service": "great"}))
eng.add_rule('high', AND({"food": "good"}, {"service": "great"}))

# %%
raw_data = {'food': [8, 5, 3, 1], 'service': [8, 5, 3, 1]}
res = eng.run_fuzz(raw_data)
for key, func in eng.input_kernel_set.items():
    print(key, func.membership_degree)
print(res)
# print(eng.actuation_signal)
# %%
# import numpy as np
# for rkey, rulelist in eng.ruleset.items():
#     print(rkey, rulelist)
#     for rule in rulelist:
#         print(np.asarray(rule(eng.input_kernel_set)))
        
    # rule_res = np.asfarray([rule(self.input_kernel_set) for rule in rulelist])
    # assert len(rule_res) >= 1, f"rule {rkey} returned no value"
