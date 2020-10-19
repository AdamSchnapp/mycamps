#!/usr/bin/env python3
import inspect
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

class FromDict:

    @classmethod
    def from_dict(self, dict):
        obj = self()
        obj.__dict__.update(dict)
        return obj


class UniqueValDict(dict):
    ''' implement key value store that prohibits repeated values'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set = set()
        for tup in self.items():
            self[tup[0]] = tup[1]

    def __setitem__(self, key, value):
        if value in self._set:
            raise ValueError(f'duplicate value {value} not allowed')
        self._set.add(value)
        super().__setitem__(key, value)

    def __delitem__(self, key):
        self._set.remove(self[key])
        super().__delitem__(key)

    def update(self, *args, **kwargs):
        if args:
            if len(args) > 1:
                raise TypeError("update expected at most 1 arguments, "
                                "got %d" % len(args))
            other = dict(args[0])
            for key in other:
                self[key] = other[key]
        for key in kwargs:
            self[key] = kwargs[key]

    def setdefault(self, key, value=None):
        if key not in self:
            self[key] = value
        return self[key]

    def flip_key_val(self):
        return {value:key for key, value in self.items()}


def removeprefix(self: str, prefix: str) -> str:
    if self.startswith(prefix):
        return self[len(prefix):]
    else:
        return self[:]

class ClassRegistry:
    ''' mixin class to enable registration of children on base_class._class_registry
        as a dictionary (key=name,value=cls); if a class is created with the same name as another,
        it will overwrite the previously registered class in _class_registry
        Note: the default_name argument is an overide to the .__name__ of the child class
        Note: warnings are raised if more than one abstract class is created '''

    def __init_subclass__(cls, default_name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls,'_class_registry'):
            cls._class_registry = dict()
        if not hasattr(cls,'_abstract_classes'):
            cls._abstract_classes = list()

        if not inspect.isabstract(cls):
            if not default_name:
                default_name = cls.__name__
            cls._class_registry[default_name] = cls
        else:
            cls._abstract_classes.append(cls)
            if len(cls._abstract_classes) == 2:
                logger.warning(f'more than one abstract classes discovered and not registered; {cls._abstract_classes}')
            if len(cls._abstract_classes) > 2:
                logger.warning(f'additional abstract classes discovered and not registered; {cls._abstract_classes[-1]}')
