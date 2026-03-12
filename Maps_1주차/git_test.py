class cal:
    def setdata(self, first, second):
        self.first = first
        self.second = second
    def add(self):
        result = self.first + self.second
        return result

    def sub(self):
        result = self.first - self.second
        return result


a = cal()
a.setdata(1,3)
print("합:", a.add())
print("차:", a.sub())