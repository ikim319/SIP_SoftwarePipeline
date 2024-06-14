%module(directors="1") numpy

%{
#include <numpy/arrayobject.h>
%}

%include "std_string.i"

%init %{
import_array();
%}
