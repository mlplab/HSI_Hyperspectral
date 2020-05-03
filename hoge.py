# coding: UTF-8


class Hoge(object):

    def __init__(self, *args, **kwargs):
        keys = kwargs.keys()
        self.name = None
        self.classes = None
        if 'name' in keys:
            self.name = kwargs['name']
        if 'classes' in keys:
            self.classes = kwargs['classes']

    def forward(self, x):
        print(self.name, x)
        print(self.classes, x)
        return self


class Fuga(Hoge):

    def __init__(self, unko, *args, chinko='chinko', **kwargs):
        super(Fuga, self).__init__(*args, **kwargs)
        self.unko = unko
        self.chinko = chinko

    def unkoward(self, x):
        print(self.name, x)
        print(self.classes, x)
        print(self.unko, x)
        print(self.chinko, x)
        return self


if __name__ == '__main__':

    hoge = Hoge()
    hoge.forward(0)
    fuga = Fuga(unko='unko', name='slsl', classes='uaua')
    fuga.forward(0)
    fuga.unkoward(1)
