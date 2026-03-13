class cal:
    first = 1 #1
    def setdata(self, _first=None, second):
        self.first = _first #2
        first = _first #3
        if _first == None:
            pass
        else:
            pass
        self.second = second
    def add(self):
        result = self.first + self.second
        return result

    def sub(self):
        result = self.first - self.second
        return result