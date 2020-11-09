#!/usr/bin/env python3
from collections import namedtuple, UserList

Step = namedtuple('Step', ['actor', 'options'])


class MultiStep(UserList):

    def __init__(self, iterable=None):
        # validation logic
        if iterable:
            [e.actor for e in iterable]
        super().__init__(iterable)

    def __setitem__(self, i, elem):
        elem.actor
        self.list[i] = elem

    def insert(self, i, elem):
        elem.actor
        super().insert(i, elem)

    def append(self, elem):
        elem.actor
        super().append(elem)

    def extend(self, iterable):
        [e.actor for e in iterable]
        super().extend(l)

    def __call__(self):
        # attempt to return lazy data
        lazy_data = None
        for step in self:
            if step.options:
                options = step.options
            else:
                options = dict()
            lazy_data = step.actor(lazy_data, **options)
        return lazy_data

    def compute(self, *args, **kwargs):
        self.computation = self()
        self.computation.compute()

    def persist(self, *args, **kwargs):
        self.computation = self()
        return self.computation.persist()
