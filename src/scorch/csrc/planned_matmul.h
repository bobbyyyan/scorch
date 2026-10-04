#pragma once

// scorch.matmul's planned fast path, in C.
//
// A repeated CSR x dense product is served by a plan (plan.h, plan.py): everything the
// ordinary dispatch would re-derive is a constant of the operand and the free dimension, so
// a planned call is the plan's checks and the kernel. What was left in front of the plan was
// the Python that finds it -- the call into a large Python function, two type tests, a key
// tuple built from b.shape and b.dtype, two dict probes, a two-level attribute read for the
// values, a call to torch.get_num_threads(), and pybind11's argument conversion into `run`.
// Warm that is a few hundred nanoseconds; cold -- after other layers have evicted the
// interpreter's objects and code -- it is a chain of dependent misses: on mkt1 a planned call
// through scorch.matmul cost 1.2-1.5 us MORE than the legacy entry point it replaced, at
// every width and on both collections, while the native plan alone costs less than either.
//
// So scorch.matmul is this function when the extension provides it. It serves exactly the
// calls the Python probe in ops.matmul served -- two positional operands, no keyword
// arguments, `type(a) is STensor`, `type(b) is torch.Tensor`, plans enabled, a plan stored
// under the same key -- and hands EVERY other call, unchanged and with the same arguments, to
// the Python function. That includes a plan that declines: the Python function then probes
// again, the plan declines again, and plan.py's decline accounting and withdrawal run exactly
// as before. Nothing about installation moves here; this only reads what plan.py wrote.
//
// The key is built to be EQUAL to the Python one, not merely analogous: ((rows, cols), dtype,
// generation). b.shape is a torch.Size, a tuple subclass that inherits tuple's hash and
// equality, so a plain tuple of the same ints finds the same entry; the dtype object is the
// interned one torch hands out for that scalar type; the generation is read from the same
// list cell the Python probe reads.
//
// fx tracing still works because tracing never reaches here: a Proxy is not an STensor, so
// the call goes to the Python function, which records `matmul` -- its module global, which
// is this object -- as the node's target. trace.py's `node.target is ops.matmul` therefore
// still holds.

#include "plan.h"

#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>

#include <string>

namespace scorch_native {

struct PlannedMatmul {
  PyObject* ordinary = nullptr;      // the Python matmul: every call a plan does not serve
  PyObject* stensor_type = nullptr;  // scorch.STensor, compared by identity like the probe
  PyObject* tensor_type = nullptr;   // torch.Tensor, likewise (a Parameter is not served)
  PyObject* generation = nullptr;    // plan.GENERATION, a one-element list
  PyObject* enabled = nullptr;       // plan.ENABLED_CELL, a one-element list
  PyObject* plans_attr = nullptr;    // interned plan.PLANS_ATTR
  PyObject* storage_attr = nullptr;  // interned "_storage"
  PyObject* value_attr = nullptr;    // interned "_value"
  bool match_host_threads = false;   // ops._MATCH_HOST_THREADS
  bool atparallel = false;           // ops._PLAN_ATPARALLEL
};

inline PlannedMatmul& planned_matmul_state() {
  static PlannedMatmul state;
  return state;
}

// The served result as a new reference; nullptr with no Python error set when this call is
// not one a plan serves (the caller then runs the ordinary path); nullptr WITH an error set
// only for a failure the ordinary path would also have raised from the same call to `run`.
inline PyObject* planned_matmul_try(const PlannedMatmul& s, PyObject* a, PyObject* b) {
  PyObject* dict = PyObject_GenericGetDict(a, nullptr);
  if (dict == nullptr) {
    PyErr_Clear();
    return nullptr;
  }
  PyObject* plans = PyDict_GetItemWithError(dict, s.plans_attr);  // borrowed; `a` owns dict
  Py_DECREF(dict);
  if (plans == nullptr || !PyDict_CheckExact(plans)) {
    PyErr_Clear();
    return nullptr;
  }
  if (!PyList_CheckExact(s.generation) || PyList_GET_SIZE(s.generation) < 1) return nullptr;

  const at::Tensor& bt = THPVariable_Unpack(b);
  const int64_t dim = bt.dim();
  PyObject* shape = PyTuple_New(dim);
  if (shape == nullptr) return nullptr;
  for (int64_t d = 0; d < dim; ++d) {
    PyObject* extent = PyLong_FromLongLong(bt.size(d));
    if (extent == nullptr) {
      Py_DECREF(shape);
      return nullptr;
    }
    PyTuple_SET_ITEM(shape, d, extent);
  }
  PyObject* key = PyTuple_Pack(3, shape, reinterpret_cast<PyObject*>(torch::getTHPDtype(bt.scalar_type())),
                               PyList_GET_ITEM(s.generation, 0));
  Py_DECREF(shape);
  if (key == nullptr) return nullptr;
  PyObject* plan_obj = PyDict_GetItemWithError(plans, key);  // borrowed
  Py_DECREF(key);
  if (plan_obj == nullptr || plan_obj == Py_None) {
    PyErr_Clear();
    return nullptr;
  }
  SpmmCsrPlan* plan = nullptr;
  try {
    plan = py::handle(plan_obj).cast<SpmmCsrPlan*>();
  } catch (const py::cast_error&) {
    return nullptr;
  }
  if (plan == nullptr) return nullptr;

  // a._storage._value, read the way the probe reads it, so a storage that computes its
  // values on access is honoured. Any failure falls through: the ordinary path raises it.
  PyObject* storage = PyObject_GetAttr(a, s.storage_attr);
  if (storage == nullptr) {
    PyErr_Clear();
    return nullptr;
  }
  PyObject* values = PyObject_GetAttr(storage, s.value_attr);
  Py_DECREF(storage);
  if (values == nullptr) {
    PyErr_Clear();
    return nullptr;
  }
  if (!THPVariable_Check(values)) {
    Py_DECREF(values);
    return nullptr;
  }
  const int64_t nthreads = s.match_host_threads ? (int64_t)at::get_num_threads() : -1;
  c10::optional<torch::Tensor> served;
  try {
    served = plan->run(THPVariable_Unpack(values), bt, nthreads, s.atparallel);
  } catch (...) {
    Py_DECREF(values);
    throw;  // translated to the Python exception by the caller's HANDLE_TH_ERRORS
  }
  Py_DECREF(values);
  if (!served.has_value()) return nullptr;
  return THPVariable_Wrap(std::move(*served));
}

inline PyObject* planned_matmul_call(PyObject* /*module*/, PyObject* const* args,
                                     Py_ssize_t nargsf, PyObject* kwnames) {
  HANDLE_TH_ERRORS
  const PlannedMatmul& s = planned_matmul_state();
  if (PyVectorcall_NARGS(nargsf) == 2 && kwnames == nullptr &&
      reinterpret_cast<PyObject*>(Py_TYPE(args[0])) == s.stensor_type &&
      reinterpret_cast<PyObject*>(Py_TYPE(args[1])) == s.tensor_type &&
      PyList_CheckExact(s.enabled) && PyList_GET_SIZE(s.enabled) >= 1 &&
      PyList_GET_ITEM(s.enabled, 0) == Py_True) {
    PyObject* served = planned_matmul_try(s, args[0], args[1]);
    if (served != nullptr || PyErr_Occurred()) return served;
  }
  return PyObject_Vectorcall(s.ordinary, args, nargsf, kwnames);
  END_HANDLE_TH_ERRORS
}

inline void planned_matmul_set(PyObject*& slot, PyObject* value) {
  Py_XINCREF(value);
  Py_XSETREF(slot, value);
}

// Builds the callable ops.py binds as `matmul`. Installing again (a module reload) rebinds the
// state; earlier callables then serve through the new state, which is what a reload means.
inline py::object install_planned_matmul(py::object ordinary, py::object stensor_type,
                                         py::object tensor_type, py::object generation,
                                         py::object enabled, const std::string& plans_attr,
                                         bool match_host_threads, bool atparallel,
                                         py::object module, const std::string& doc) {
  if (!PyCallable_Check(ordinary.ptr())) throw py::type_error("ordinary must be callable");
  if (!PyType_Check(stensor_type.ptr()) || !PyType_Check(tensor_type.ptr()))
    throw py::type_error("stensor_type and tensor_type must be types");
  if (!PyList_CheckExact(generation.ptr()) || PyList_GET_SIZE(generation.ptr()) < 1 ||
      !PyList_CheckExact(enabled.ptr()) || PyList_GET_SIZE(enabled.ptr()) < 1)
    throw py::type_error("generation and enabled must be one-element lists");
  PlannedMatmul& s = planned_matmul_state();
  planned_matmul_set(s.ordinary, ordinary.ptr());
  planned_matmul_set(s.stensor_type, stensor_type.ptr());
  planned_matmul_set(s.tensor_type, tensor_type.ptr());
  planned_matmul_set(s.generation, generation.ptr());
  planned_matmul_set(s.enabled, enabled.ptr());
  Py_XSETREF(s.plans_attr, PyUnicode_InternFromString(plans_attr.c_str()));
  if (s.storage_attr == nullptr) s.storage_attr = PyUnicode_InternFromString("_storage");
  if (s.value_attr == nullptr) s.value_attr = PyUnicode_InternFromString("_value");
  if (s.plans_attr == nullptr || s.storage_attr == nullptr || s.value_attr == nullptr)
    throw py::error_already_set();
  s.match_host_threads = match_host_threads;
  s.atparallel = atparallel;

  // The method table outlives every function object made from it. The docstring is kept from
  // the first install: a function object holds a pointer into it.
  static std::string doc_text = "matmul(a, b, /, **kwargs)\n--\n\n" + doc;
  static PyMethodDef def = {
      "matmul",
      reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)(void)>(planned_matmul_call)),
      METH_FASTCALL | METH_KEYWORDS, doc_text.c_str()};
  // `self` is the module, so the function reports __qualname__ "matmul" and pickles by
  // reference as module.matmul, exactly as the Python function did.
  PyObject* name = PyObject_GetAttrString(module.ptr(), "__name__");
  if (name == nullptr) throw py::error_already_set();
  PyObject* fn = PyCFunction_NewEx(&def, module.ptr(), name);
  Py_DECREF(name);
  if (fn == nullptr) throw py::error_already_set();
  return py::reinterpret_steal<py::object>(fn);
}

}  // namespace scorch_native
