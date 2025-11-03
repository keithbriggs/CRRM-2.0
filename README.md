CRRM 5G system-level simulator
------------------------------

Keith Briggs and Ibrahim Nur

The CRRM (cellular radio reference model) simulator emulates a cellular radio system following 5G concepts and channel models. The intention is to have an easy-to-use, scalable, and very fast system written in pure Python with minimal dependencies. 

Note that CRRM is a *system-level* simulator, not a link-level simulator. This means that it takes a coarse-grained approach, specifically meaning that it does not model concepts like packet flows and queueing at all. Resource allocation is modelled, but only as a continuous process that ignores discrete resource blocks. These simplifications are necessary if large systems are to be simulated. The main application areas are the evaluation of high-level network management strategies, not the accurate estimation of throughputs to indvidual devices. Other software is available for that type of link-level simulation.

The simulator builds on an earlier one developed for the AIMM project (<https://github.com/keithbriggs/AIMM-simulator>) by Keith Briggs (<https://keithbriggs.info>), but is a complete rewrite with many improvements. It also uses ideas from the CRM project by Kishan Sthankiya (<https://github.com/apw804/CellularReferenceModel>).

Software dependencies
---------------------

1. Python 3.8 or higher <https://python.org>.
2. NumPy <https://numpy.org/>. 
3. SciPy <https://scipy.org/>. 
4. Matplotlib <https://matplotlib.org>.

Installation from PyPi
-----------------------

pip install CRRM

Installation from source
------------------------

1. Download zip from the green "<> Code" tab above
2. unzip CRRM-2.0.zip
3. cd CRRM-2.0
4. pip install .

Documentation
-------------

See <https://crrm-20.readthedocs.io/en/latest/>.

Local version in doc/sphinx-build/index.html.
