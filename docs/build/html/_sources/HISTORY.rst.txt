=======
History
=======

0.1.0 (2022–01–28)
— — — — — — — — — —

* Added Linguistic and Sugeno-Takagi Deffuzification methods
* Added Engine.gen_surface
* Library now runs on top of numpy, for vectorized math optimizations
* TODO: caching results and gen_surface usage, for quick calculations
 

0.0.1 (2022–01–18)
— — — — — — — — — —

* First release on PyPI.
* Created Engine, Kernel, Rules, Operand and Rules modules
* Library builds the Engine and map data input to fuzzy output, according to Kernel, Rules and Operands
* TODO: missing an Engine.defuzzyfy method.
* TODO: missing an Engine.generate surface method, for caching full set calculations
 