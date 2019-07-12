from sympy import Function, Symbol

class Control:
    def __init__(self, symbol, type):
        assert type in ['Voltage', 'Flux'], 'Unrecognized control type'
        self.symbol = symbol
