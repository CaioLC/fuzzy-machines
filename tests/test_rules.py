from fuzzy_machines.operand import OperandEnum
from fuzzy_machines.engine import Engine
from fuzzy_machines.memb_funcs import Constant, Linear
from fuzzy_machines.kernel import Kernel, KernelFuncMember
from fuzzy_machines.rules import AND, OR, NOT

food = (Kernel(0, 10)
            .add_memb_func('good', KernelFuncMember(Linear(.1, 0)))
            .add_memb_func('rancid', KernelFuncMember(Linear(-.1, 1)))
)
food(8)

service = (Kernel(0, 10)
            .add_memb_func('good', KernelFuncMember(Linear(.1, 0)))
            .add_memb_func('bad', KernelFuncMember(Linear(-.1, 1)))
)
service(3)

price = (Kernel(0, 10)
            .add_memb_func('cheap', KernelFuncMember(Linear(-.1, 1)))
            .add_memb_func('expensive', KernelFuncMember(Linear(.1, 0)))
)
price(7)

input_kernel_set = {
    'food': food.input_membership,
    'service': service.input_membership,
    'price': price.input_membership,
}

def test_and():
    op = AND({'food': 'good'}, {'service': 'good'}, OperandEnum.DEFAULT)
    assert round(op(input_kernel_set),1) == .3

def test_or():
    op = OR({'food': 'good'}, {'service': 'good'}, OperandEnum.DEFAULT)
    assert round(op(input_kernel_set),1) == .8

def test_not():
    op = NOT({'food': 'good'}, OperandEnum.DEFAULT)
    assert round(op(input_kernel_set),1) == .2

def test_nested():
    op = OR(
        AND({'food':'good'}, {'price':'cheap'}, OperandEnum.DEFAULT),
        AND({'food': 'rancid'}, {'service': 'good'}, OperandEnum.DEFAULT),
        OperandEnum.DEFAULT
    )
    assert round(op(input_kernel_set), 1) == .3

    op = OR(
            AND(
                {'food':'good'},
                AND({'service': 'good'}, {'price': 'expensive'},OperandEnum.DEFAULT),
                OperandEnum.DEFAULT),
            AND(
                {'food': 'rancid'},
                AND({'service': 'good'}, NOT({'price': 'expensive'},OperandEnum.DEFAULT),OperandEnum.DEFAULT),
                OperandEnum.DEFAULT),
            OperandEnum.DEFAULT
    )
    assert round(op(input_kernel_set), 1) == .3
