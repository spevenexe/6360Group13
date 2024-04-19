class Parameters:
    def __init__(self):
        self.params = {}

    def set(self, name, value):
        self.params[name] = str(value)

    def get(self, name, default=None):
        return self.params.get(name, default)