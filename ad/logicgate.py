class LogicGate:

    def __init__(self, n):
        self.label = n
        self.output = None

    def getLabel(self):
        return self.label

    def getOutput(self):
        self.output = self.performGateLogic()
        return self.output


class BinaryGate(LogicGate):

    def __init__(self, n):
        LogicGate.__init__(self, n)

        self.pinA = None
        self.pinB = None

    def getPinA(self):
        if self.pinA == None:
            pin = int(input("Enter Pin A input for binary-gate " + self.getLabel() + "-->"))
            if pin not in [0, 1]:
                raise TypeError('Not a signal in {0, 1}')
            else:
                return pin
        else:
            return self.pinA.getFrom().getOutput()

    def getPinB(self):
        if self.pinB == None:
            pin = int(input("Enter Pin B input for binary-gate " + self.getLabel() + "-->"))
            if pin not in [0, 1]:
                raise TypeError('Not a signal in {0, 1}')
            else:
                return pin
        else:
            return self.pinB.getFrom().getOutput()

    def setNextPin(self, source):
        if self.pinA == None:
            self.pinA = source
        else:
            if self.pinB == None:
                self.pinB = source
            else:
                raise RuntimeError("Error: NO EMPTY PINS")


class AndGate(BinaryGate):

    def __init__(self, n):
        BinaryGate.__init__(self, n)

    def performGateLogic(self):

        a = self.getPinA()
        b = self.getPinB()
        if a == 1 and b == 1:
            return 1
        else:
            return 0


class OrGate(BinaryGate):

    def __init__(self, n):
        BinaryGate.__init__(self, n)

    def performGateLogic(self):

        a = self.getPinA()
        b = self.getPinB()
        if a == 1 or b == 1:
            return 1
        else:
            return 0


class XorGate(BinaryGate):

    def __init__(self, n):
        BinaryGate.__init__(self, n)

    def performGateLogic(self):

        a = self.getPinA()
        b = self.getPinB()
        if a + b == 1:
            return 1
        else:
            return 0


class UnaryGate(LogicGate):

    def __init__(self, n):
        LogicGate.__init__(self, n)

        self.pin = None

    def getPin(self):
        if self.pin == None:
            pin = int(input("Enter Pin input for unary-gate " + self.getLabel() + "-->"))
            if pin not in [0, 1]:
                raise TypeError('Not a signal in {0, 1}')
            else:
                return pin
        else:
            return self.pin.getFrom().getOutput()

    def setNextPin(self, source):
        if self.pin == None:
            self.pin = source
        else:
            raise RuntimeError("Cannot Connect: NO EMPTY PINS on this gate")


class NotGate(UnaryGate):

    def __init__(self, n):
        UnaryGate.__init__(self, n)

    def performGateLogic(self):
        if self.getPin():
            return 0
        else:
            return 1


class Connector:

    def __init__(self, fgate, tgate):
        self.fromgate = fgate
        self.togate = tgate

        tgate.setNextPin(self)

    def getFrom(self):
        return self.fromgate

    def getTo(self):
        return self.togate


class Halfadder(BinaryGate):

    def __init__(self, n):
        BinaryGate.__init__(self, n)

    def performGateLogic(self):
        a = self.getPinA()
        b = self.getPinB()

        sum = 1 if a + b == 1 else 0
        carry = 1 if a + b == 2 else 0

        return sum, carry


def main():
    g1 = AndGate("G1")
    g2 = AndGate("G2")
    g3 = OrGate("G3")
    g4 = NotGate("G4")
    c1 = Connector(g1, g3)
    c2 = Connector(g2, g3)
    c3 = Connector(g3, g4)
    print(g4.getOutput())

# main()
