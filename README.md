# OptiVLSI

*Very Large Scale Integration abbreviated as VLSI involves digital circuits dealing with billions of transistors, and
requires computerized design automation, design verification and testing algorithms. Digital circuits are often
represented using graphs, where the logic gates are nodes and their interconnections are the edges. Since VLSI
circuits have millions of logic gates, there is a need for fast and highly optimized graph algorithms. There are a
few optimized graph libraries with optimized basic graph operations such as `networkx`, but there are no optimized
and high performance open source libraries which build upon these to specifically cater to the VLSI computer
aided design automation industry.*


## Initially Implemented Optimised Implementations/Algorithms (Documentation can be found [here](https://github.com/rohankalbag/optiVLSI/blob/main/OptiVLSI.pdf))

- [Lee Algorithm](https://github.com/rohankalbag/optiVLSI/tree/main/lee-algorithm)
- [Kruskal's Algorithm](https://github.com/rohankalbag/optiVLSI/tree/main/kruskal)
- [Binary Decision Diagrams](https://github.com/rohankalbag/optiVLSI/tree/main/ROBDD)
- [Bellman-Ford Algorithm](https://github.com/rohankalbag/optiVLSI/tree/main/bellman-ford)
- [Prim Algorithm](https://github.com/rohankalbag/optiVLSI/tree/main/prim)
- [Dijkstra's Algorithm](https://github.com/rohankalbag/optiVLSI/tree/main/dijkstra)
- [Compiled Code Simulator](https://github.com/rohankalbag/optiVLSI/tree/main/compiled-code-simulator)
- [Event Driven Simulator](https://github.com/rohankalbag/optiVLSI/tree/main/event-driven-sim)

### Optimization Tools Used
- Numba: All the algorithms that are implemented have been accelerated using numba and their runtimes have
been compared with pythonic and other implementations
- Automan: To automate simulations, benchmark algorithms, several circuits/graphs of varying sizes were used
and results for the same were generated using automan


##### This open-source codebase started as a course project for the course offered at IIT Bombay, AE6102 - Parallel Scientific Computing and Visualization. We are open to more open-source contributions

### Collaborators
- Rohan Rajesh Kalbag
- Neeraj Prabhu
