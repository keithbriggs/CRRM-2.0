CRRM 5G system-level simulator
------------------------------

Keith Briggs and Ibrahim Nur

The CRRM (cellular radio reference model) simulator emulates a cellular radio system following 5G concepts and channel models. The intention is to have an easy-to-use, scalable, and very fast system written in pure Python with minimal dependencies. It is especially designed to be suitable for interfacing to AI engines such as ``tensorflow`` or ``pytorch``.  The simulator builds on an earlier one developed for the AIMM project (<https://github.com/keithbriggs/AIMM-simulator>) by Keith Briggs (<https://keithbriggs.info>), but is a complete rewrite with many improvements. It also uses ideas from the CRM project by Kishan Sthankiya (<https://github.com/apw804/CellularReferenceModel>).

Software dependencies
---------------------

1. Python 3.8 or higher <https://python.org>.
2. NumPy <https://numpy.org/>. 
3. SciPy <https://scipy.org/>. 
4. Matplotlib <https://matplotlib.org>.

Installation from source
-----------------------

1. unzip CRRM-2.0.zip
2. cd CRRM-2.0
3. pip install .

Documentation
-------------

See <https://crrm-20.readthedocs.io/en/latest/>.

Local version in doc/sphinx-build/index.html.
