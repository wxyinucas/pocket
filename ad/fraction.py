###########################
#
# http://interactivepython.org/runestone/static/pythonds/Introduction/ObjectOrientedProgramminginPythonDefiningClasses.html
#
#
###########################


class Fraction:

    def __init__(self, top, bottom):

        if type(top) != 'int' or type(bottom) != 'int':
            raise RuntimeError('Not int!!')
        common = gcd(top, bottom)
        self.num = top // common
        self.den = bottom // common

    def show(self):
        print(self.num, '/', self.den)

    def __str__(self):
        return str(self.num) + '/' + str(self.den)

    def __add__(self, other):
        newnum = self.num * other.den + other.num * self.den
        newden = self.den * other.den

        return Fraction(newnum, newden)

    def __eq__(self, other):
        firstnum = self.num * other.den
        secondnum = other.num * self.den

        return firstnum == secondnum

    def getNum(self):
        return self.num

    def getDen(self):
        return self.den


def gcd(m, n):
    if m < n:
        temp = n
        n = m
        m = temp

    while m % n != 0:
        oldm = m
        oldn = n

        m = oldn
        n = oldm % oldn

    return n
