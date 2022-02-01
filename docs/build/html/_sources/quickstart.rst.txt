.. highlight:: shell

===========
Quick Start
===========

**Fuzzy Machines** is a Python library for building general fuzzy logic inference systems.

This chapters is intended as a quick and simple example that should give you the big picture on how to use the library, but it is 
not intended as a full explanation on how inference systems work, or why use them. For further reference on the subject as a whole, there is a great youtube series at https://www.youtube.com/watch?v=__0nZuG4sTw. As 
a matter of fact, in this tutorial we will be building a fuzzy machine inspired by the "Restaurant Tip" example of the series, so you can see what a python implementation using this library would look like.

First Step: defining the problem
--------------------------------

Designing fuzzy logic inference systems can get pretty complicated pretty rapidly. So before we get our hands dirty, it is always best to have a clear understanding of what is our data and what we want to achieve.

Our system is a simple one: we want to define a fair tip percentage we give at a restaurant, given the food quality and service quality. Our data for food and service quality ranges from 0-10 and we want to give tips between 10% and 30%.

Second Step: building the Input Kernels
---------------------------------------

The Kernel is one of the building blocks of the Engine we will create following this tutorial. Each Kernel is responsible for mapping raw data about a particular variable of interest to various inner functions. 
In our case, we need one Kernel to describe food quality and a second one to describe food service.

Each Kernel is comprised of inner membership functions of type MembershipFunction. If a Kernel describe a particular variable of interest, each MembershipFunction describe a particular state of such variable.
For instance: (i) the 'food' variable could be 'excelent' or 'rancid'; and (ii) the 'service' variable could be 'great' or 'poor'.

For each state ('good', 'rancid', 'great' and 'poor') we will instatiate the MembershipFunction that maps the raw data (remember, 0-10 for both variables, but it will vary from case to case) to a particular function.
For consistency at the API level, instead of passing a generic function directly, we will instantiate with MembershipFunction (any class that inherits from MembershipFunction).

Putting it all together we have:

.. code-block:: python3

    from fuzzy_machines.kernel import Kernel
    from fuzzy_machines.memb_funcs import Constant, Linear

    food = Kernel(0, 10) # instatiate the "Food Quality Kernel" and register that raw data ranges from 0 to 10 (inclusive)
    food.add_memb_func("good", Linear(0, 10)) # register a linear membership function defining what is a 'good' food quality
    food.add_memb_func("rancid", Linear(10, 0)) # register a linear membership function defining what is a 'rancid' food quality

    # now we do the same for the service quality kernel:
    service = Kernel(0, 10)
    service.add_memb_func("great", Linear(0, 10))
    service.add_memb_func("poor", Linear(10, 0))

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

    tips = Kernel(10, 30) # tips will range between 10% and 30%
    tips.add_memb_func("low", Constant(10)) # if low we give 10% tip to the restaurant
    tips.add_memb_func("average", Constant(20))
    tips.add_memb_func("high", Constant(30))


Fourth Step: Rules
------------------
This is pretty self explanatory. We want to add declarations on how to map food and service quality to the amount of tip we pay the restaurant. This is done by adding rules to the engine, which are of type RuleBase.
In our example, the tip will be low if food quality is rancid. Average if food quality is good but service is poor. And high if food is good and service is great:

.. code-block:: python3

    from fuzzy_machines.rules import AND, OR, IS, NOT

    low = {"low": IS({"food": 'rancid'})}
    average = {"average": AND({"food": "good"}, {"service": "poor"})}
    high = {"high": AND({"food": "good"}, {"service": "great"})}


Fifth Step: Putting it all together
-----------------------------------
Now it is time to fire up the engine. We create a new Engine object and register the input kernels, inference system and rules: 

.. code-block:: python3

    from fuzzy_machines.engine import Engine

    eng = (
    Engine()
        .add_kernel("food", food)
        .add_kernel("service", service)
        .add_inference_kernel(tips)
        .add_rule(low)
        .add_rule(average)
        .add_rule(high)
    )

Running the machine
-------------------
With all in place, all you now need to do is fire up the engine. Call eng.run_fuzz() or eng.run_defuzz() with the raw data for food and service quality, and you should get the corresponding fuzzy result and defuzzy output (respectively) for the tips amount.

.. code-block:: python3

    raw_data_example = {'food': 9}, {'service': 3}
    fuzzy_results = eng.run_fuzz(raw_data_example)
    print(fuzzy_results)
    ## OR ##
    defuzzy_results = eng.run_defuzz(raw_data_example)
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
