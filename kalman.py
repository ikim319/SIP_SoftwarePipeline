# This file was automatically generated by SWIG (https://www.swig.org).
# Version 4.2.1
#
# Do not make changes to this file unless you know what you are doing - modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _kalman
else:
    import _kalman

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "this":
            set(self, name, value)
        elif name == "thisown":
            self.this.own(value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


import weakref

class KalmanFilter(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, state_dim, meas_dim):
        _kalman.KalmanFilter_swiginit(self, _kalman.new_KalmanFilter(state_dim, meas_dim))

    def predict(self):
        return _kalman.KalmanFilter_predict(self)

    def update(self, z):
        return _kalman.KalmanFilter_update(self, z)

    def setStateTransition(self, F):
        return _kalman.KalmanFilter_setStateTransition(self, F)

    def setMeasurementMatrix(self, H):
        return _kalman.KalmanFilter_setMeasurementMatrix(self, H)

    def setProcessNoiseCovariance(self, Q):
        return _kalman.KalmanFilter_setProcessNoiseCovariance(self, Q)

    def setMeasurementNoiseCovariance(self, R):
        return _kalman.KalmanFilter_setMeasurementNoiseCovariance(self, R)

    def setInitialState(self, x0):
        return _kalman.KalmanFilter_setInitialState(self, x0)

    def setInitialCovariance(self, P0):
        return _kalman.KalmanFilter_setInitialCovariance(self, P0)

    def getState(self):
        return _kalman.KalmanFilter_getState(self)

    def getCovariance(self):
        return _kalman.KalmanFilter_getCovariance(self)
    __swig_destroy__ = _kalman.delete_KalmanFilter

# Register KalmanFilter in _kalman:
_kalman.KalmanFilter_swigregister(KalmanFilter)
