.. highlight:: shell

===========
Quick Start
===========

**Fuzzy Machines** is a Python library for building general fuzzy logic inference systems.

This chapters is intended as a quick and simple example that should give you the big picture on how to use the library, but it is 
not intended as a full explanation on how inference systems work, or why use them. 

For further reference on the subject as a whole, there is a great youtube series at https://www.youtube.com/watch?v=__0nZuG4sTw. As 
a matter of fact, in this tutorial we will be building a fuzzy machine inspired by the "Restaurant Tip" example of the series, so you can see what a python implementation using this library would look like.

First Step: defining the problem
--------------------------------

Designing fuzzy logic inference systems can get pretty complicated pretty rapidly. So before we get our hands dirty, it is always best to have a clear understanding of what is our data and what we want to achieve.

Our system is a simple one: we want to define a fair tip percentage we give at a restaurant, given the food quality and service quality. Our data for food and service quality ranges from 0-10 and we want to give tips between 10% and 30%.

Second Step: building the Input Kernels
---------------------------------------

The Kernel is one of the building blocks of the Engine we will create following this tutorial. Each Kernel is responsible for mapping raw data about a particular variable of interest to various inner functions. 
In our case, we need one Kernel to describe food quality and a second one to describe food service.

Each Kernel is comprised of inner MembershipFunctions. If a Kernel describe a particular variable of interest, each MembershipFunction describe a particular state of such variable.
For instance: the variable 'food' can be categorized as 'good' or 'rancid'; the variable 'service' could be 'great' or 'poor'.

We create a Kernel object and, for each state ('good', 'rancid', 'great' and 'poor') we will instatiate the appropriate MembershipFunction that maps the raw data to a particular variable. For simplicity, all 
MembershipFunctions will be of type Linear, where the first parameter marks where y == 0 and the second marks where y == 1, for any raw data that is passed by the user. Hence:

.. code-block:: python3

    from fuzz.kernel import Kernel
    from fuzz.memb_funcs import Linear

    food = Kernel(0, 10) # instatiate the "Food Quality Kernel" and register that raw data ranges from 0 to 10 (inclusive)
    food.add_memb_func("good", Linear(4, 10)) # register a linear membership function defining what is a 'good' food quality
    food.add_memb_func("rancid", Linear(8, 4)) # register a linear membership function defining what is a 'rancid' food quality

    # now we do the same for the service quality kernel:
    service = Kernel(0, 10)
    service.add_memb_func("great", Linear(5, 10))
    service.add_memb_func("poor", Linear(7, 3))

Note: **Fuzzy Machines** implements the *builder pattern*, which is a creational design pattern that lets you construct complex objects step by step. You can always create an "empty" Kernel object
(or an empty Engine object as we will soon see) and incrementally add the necessary blocks. As each building block returns 'self', you could also pipe all method calls when instatiating object. 
In this case, creating the food and service kernels would be:

.. code-block:: python3

    food = (
        Kernel(0,10)
            .add_memb_func('good', Linear(4, 10))
            .add_memb_func('rancid', Linear(8, 4))
    )

    service = (
        Kernel(0,10)
            .add_memb_func('great', Linear(5, 10))
            .add_memb_func('poor', Linear(7, 3))
    )


.. warning::
    By definition, membership functions maps how much the raw data "fits" each the state definition. The same way probability ranges only between 0-1 (0-100%),
    the Kernel clamps all function results to be between 0 and 1. If the user register a MembershipFunction that returns values greater than 1 or less than 0, all
    such results will be transformed to 1 and 0 respectively. A quick way to check if the MembershipFunction are properly set is calling the kernel.describe method and ploting
    the results with your preferred plotting library.

Third Step: Inference System
----------------------------
The inference system is also a Kernel, which is formed by MembershipFunctions, so the building process is the same as above. The inference system maps the fuzzy states of the output we're interest.
In our case, we can give the restaurant a 'low', 'average' or 'high' tip, depending on food and service quality. Thus:

.. code-block:: python3

    tips = (
        Kernel(10,30)
            .add_memb_func('low', Constant(10,15))
            .add_memb_func('average', Constant(15,25))
            .add_memb_func('high', Constant(25,30))
    )

Fourth Step: Defining the Engine
--------------------------------

With the Input Kernels and Inference System defined, we can now instatiate our Engine.

.. code-block:: python3

    from fuzz.engine import Engine

    eng = (
        Engine()
            .add_kernel("food", food)
            .add_kernel("service", service)
            .add_inference_kernel(tips)
    )

Fifth Step: Setting the Rules
-----------------------------

This is the last step of wiring. Basically, we want to add declarations that map food and service quality to the amount of tip we pay the restaurant. This is done by adding rules to the engine, which are dictionaries mapping a RuleBase to a particular MembershipFunction of the Inference System.

In our example, we have 'low', 'average' and 'high' MembershipFunctions of the 'tips' Inference System. And we want the tip to be low if food quality is rancid. Average if food quality is good but service is poor. And high if food is good and service is great:

.. code-block:: python3

    from fuzz.rules import AND, OR, IS, NOT

    eng.add_rule('low', IS({'food':'rancid'}))
    eng.add_rule('average', AND({"food": "good"}, {"service": "poor"}))
    eng.add_rule('high', AND({"food": "good"}, {"service": "great"}))

Running the machine
-------------------
With all in place, all you now need to do is fire up the engine. Suppose we just visited the restaurant and rated the food to be 8 (from 0 to 10), but service wasn't so good (4 out of 10).
Call eng.run_fuzz() or eng.run_defuzz() with the raw data for food and service quality, and you should get the corresponding fuzzy result and defuzzy output (respectively) for the tips amount.

.. code-block:: python3

    raw_data = {'food': 8, 'service': 3}
    fuzzy_results = eng.run_fuzz(raw_data)
    print(fuzzy_results) ## {'low': array(0.), 'average': array(0.66666667), 'high': array(0.)}

    ## OR ##
    defuzzy_results = eng.run_defuzz(raw_data)
    print(defuzzy_results)

TL;DR
---------------------------------
Here's what we we need to do for any fuzzy machine:

1. Define the problem: what is the raw data input, the variables, states and inference system and rules.
2. Build the Kernels for each input variable
3. Build the Kernel for the inference system
4. Declare the rules, mapping the kernel input to the inference system
5. Register everything at the Engine level (register Kernels, Inf. System and rules)
6. Fire up the engine with the raw data you have at hands. Get the fuzzy or defuzzy output


Here's the full sample code: 

.. code-block:: python3

    from fuzzy_machines.engine import Engine
    from fuzzy_machines.kernel import Kernel
    from fuzzy_machines.memb_funcs import Constant, Linear
    from fuzzy_machines.rules import AND, OR, NOT

    # Input Kernels:
    # a. Food Kernel
    food = Kernel(0, 10) # instatiate the "Food Quality Kernel" and register that raw data ranges from 0 to 10 (inclusive)
    food.add_memb_func("good", Linear(0, 10)) # register a MembershipFunction for what is a 'good' food quality
    food.add_memb_func("rancid", Linear(10, 0)) # register a MembershipFunction for what is a 'rancid' food quality

    # b. Service Kernel
    service = Kernel(0, 10)
    service.add_memb_func("great", Linear(0, 10))
    service.add_memb_func("poor", Linear(10, 0))

    # Inference System:
    tips = Kernel(10, 30) # tips will range between 10% and 30%
    tips.add_memb_func("low", Constant(10)) # if low we give 10% tip to the restaurant
    tips.add_memb_func("average", Constant(20))
    tips.add_memb_func("high", Constant(30))

    # Rules:
    low = {"low": IS({"food": 'rancid'})}
    average = {"average": AND({"food": "good"}, {"service": "poor"})}
    high = {"high": AND({"food": "good"}, {"service": "great"})}

    # Putting it all together
    eng = (
    Engine()
        .add_kernel("food", food)
        .add_kernel("service", service)
        .add_inference_kernel(tips)
        .add_rule(low)
        .add_rule(average)
        .add_rule(high)
    )

    # Fire the engine
    raw_data_example = {'food': 9}, {'service': 3}
    fuzzy_results = eng.run_fuzz(raw_data_example)
    print(fuzzy_results)
    ## OR ##
    defuzzy_results = eng.run_defuzz(raw_data_example)
    print(defuzzy_results)
