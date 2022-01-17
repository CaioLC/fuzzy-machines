from types import FunctionType
from fuzzy_machines.kernel import Kernel, KernelFuncMember, _clamp
from fuzzy_machines.memb_funcs import FunctionBase, Constant, Linear

def test_clamp():
    assert 0 == _clamp(-1.234, 0, 1), "returned value below lower boundary"
    assert 1 == _clamp(1.234, 0, 1), "returned value above higher boundary"
    assert .234 == _clamp(.234, 0, 1), "did not returned value between boundaries"

def test_kernel_func_member_init():
    kmember = KernelFuncMember(Constant(.5))
    assert isinstance(kmember.func, FunctionBase)

def test_kernel_func_member_call():
    kmember = KernelFuncMember(Constant(.5))
    assert kmember(20) == .5
    assert kmember('string') == .5
    assert kmember([20, 10]) == .5
    assert kmember({'20':20}) == .5
    assert kmember((20, 'v')) == .5

    kmember = KernelFuncMember(Linear(.1, 0))
    assert kmember(0) == 0
    assert kmember(1) == .1
    assert kmember(5) == .5
    assert kmember(10) == 1
    assert kmember(50) == 1
    assert kmember(-1) == 0
    try:
        kmember('str')
    except TypeError:
        pass

def test_kernel_func_member_iterate():
    kmember = KernelFuncMember(Linear(.5, 0))
    points = kmember.iterate(-1, 1, 5)
    assert [0,0,0,.25, .5] == points

def test_kernel_init():
    pass

def test_kernel_add_func():
    food = Kernel(0, 10)
    food.add_memb_func('good', KernelFuncMember(Linear(.1, 0)))
    assert isinstance(food.input_functions, dict)
    assert food.input_functions['good']
    food.add_memb_func('bad', KernelFuncMember(Linear(-.1, 0)))
    assert isinstance(food.input_functions, dict)
    assert food.input_functions['good']
    assert food.input_functions['bad']
    assert len(food.input_functions) == 2
    try:
        food.input_functions['error'] # should fail
    except KeyError:
        pass

    assert isinstance(food.input_functions['good'], KernelFuncMember)
    assert isinstance(food.input_functions['bad'], KernelFuncMember)

    def regular_func_fail():
        return 0.5
    try:
        food.add_memb_func('func_error', regular_func_fail) # should fail
    except TypeError:
        pass

def test_kernel_del_func():
    food = Kernel(0, 10)
    food.add_memb_func('good', KernelFuncMember(Linear(.1, 0)))
    food.add_memb_func('bad', KernelFuncMember(Linear(-.1, 0)))
    food.del_memb_func('good')
    assert len(food.input_functions) == 1
    assert food.input_functions.keys() == set(['bad'])
    food.del_memb_func('bad')
    assert len(food.input_functions) == 0
    assert food.input_functions.keys() == set([])
    assert isinstance(food.input_functions, dict)
    try:
        food.del_memb_func('error') # should fail
    except KeyError:
        pass

def test_kernel_call():
    food = Kernel(0, 10)
    food.add_memb_func('good', KernelFuncMember(Linear(.1, 0)))
    food.add_memb_func('bad', KernelFuncMember(Linear(-.1, 1)))
    print(food.input_membership)
    food(8)
    assert round(food.input_membership['good'], 1) == 0.8
    assert round(food.input_membership['bad'], 1) == 0.2

def test_kernel_describe():
    food = Kernel(0, 10)
    food.add_memb_func('good', KernelFuncMember(Linear(.1, 0)))
    food.add_memb_func('bad', KernelFuncMember(Linear(-.1, 1)))
    res = food.describe(11)
    assert [round(val,1) for val in res['good']] == [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
    assert [round(val,1) for val in res['bad']] == [1.,.9,.8,.7,.6,.5,.4,.3,.2,.1,0]

    food = Kernel(-5, 15)
    food.add_memb_func('good', KernelFuncMember(Linear(.1, 0)))
    food.add_memb_func('bad', KernelFuncMember(Linear(-.1, 1)))
    res = food.describe(21)
    assert [round(val,1) for val in res['good']] == [0, 0, 0, 0, 0, 0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1,1,1,1,1,1]
    assert [round(val,1) for val in res['bad']] == [1,1,1,1,1,1.,.9,.8,.7,.6,.5,.4,.3,.2,.1,0,0,0,0,0,0]
