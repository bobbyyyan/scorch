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
//
// The left operand may also be a torch sparse CSR tensor (`type(a) is torch.Tensor` with the
// sparse CSR layout), whose plans plan.py keeps in plan.TORCH_OPERANDS, keyed on id(a) and
// checked against a weak reference, because a torch tensor cannot carry them (see there).
// Its values and index arrays are read from the tensor's implementation, so a served call
// creates no Python object, and the index arrays go to `run_csr`, which declines unless they
// are still the ones the plan was built from. The layout test is a field read on the
// TensorImpl, so a dense left operand -- every dense matmul made through `import scorch as
// torch` -- pays one comparison for it before going to the Python function as before.
//
// For a torch operand this path also owns the SIGHTINGS (plan.install's rule that a plan is
// installed only for an operand seen before). An operand built afresh for every call is only
// ever sighted, never planned, so whatever a sighting costs is what such a call site pays for
// plans existing at all: done in Python at the end of the ordinary path it was 2 us of a mkt1
// call. So an unserved torch operand goes to the Python function with the key under a private
// keyword (ops._C_PROBED_KW), which tells it the probe was done here and the sighting will be
// recorded here, once it returns. Python keeps installation and decline accounting.

#include "plan.h"

#include <ATen/SparseCsrTensorUtils.h>
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
  // Torch sparse CSR operands: their plans and sightings live in plan.TORCH_OPERANDS.
  PyObject* torch_operands = nullptr;    // plan.TORCH_OPERANDS: id -> (weak reference, state)
  PyObject* operand_ref_type = nullptr;  // plan._OperandRef, the weak reference an entry holds
  PyObject* operand_died = nullptr;      // plan._drop_torch_operand, its callback
  PyObject* seen_attr = nullptr;         // interned plan.SEEN_ATTR
  PyObject* key_attr = nullptr;          // interned "key", _OperandRef's slot
  PyObject* probed_kwnames = nullptr;    // (ops._C_PROBED_KW,): the keyword the key travels under
  Py_ssize_t max_plans = 0;              // plan.MAX_PLANS_PER_OPERAND
  bool match_host_threads = false;       // ops._MATCH_HOST_THREADS
  bool atparallel = false;               // ops._PLAN_ATPARALLEL
};

inline PlannedMatmul& planned_matmul_state() {
  static PlannedMatmul state;
  return state;
}

// The key a plan is stored under, ((b rows, b cols), b dtype, generation), as a new reference;
// nullptr with an error set if it cannot be built.
inline PyObject* planned_matmul_key(const PlannedMatmul& s, const at::Tensor& bt) {
  if (!PyList_CheckExact(s.generation) || PyList_GET_SIZE(s.generation) < 1) {
    PyErr_SetString(PyExc_RuntimeError, "plan generation cell is malformed");
    return nullptr;
  }
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
  return key;
}

inline SpmmCsrPlan* planned_matmul_cast(PyObject* plan_obj) {
  try {
    return py::handle(plan_obj).cast<SpmmCsrPlan*>();
  } catch (const py::cast_error&) {
    return nullptr;
  }
}

// The plan stored in `plans` under `key`, or nullptr with no error set when there is none.
inline SpmmCsrPlan* planned_matmul_find(PyObject* plans, PyObject* key) {
  if (plans == nullptr || !PyDict_CheckExact(plans)) return nullptr;
  PyObject* plan_obj = PyDict_GetItemWithError(plans, key);  // borrowed
  if (plan_obj == nullptr || plan_obj == Py_None) {
    PyErr_Clear();
    return nullptr;
  }
  return planned_matmul_cast(plan_obj);
}

inline int64_t planned_matmul_nthreads(const PlannedMatmul& s) {
  return s.match_host_threads ? (int64_t)at::get_num_threads() : -1;
}

// An STensor operand. The served result as a new reference; nullptr with no Python error set
// when this call is not one a plan serves (the caller then runs the ordinary path); nullptr
// WITH an error set only for a failure the ordinary path would also have raised from the same
// call to `run`.
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

  const at::Tensor& bt = THPVariable_Unpack(b);
  PyObject* key = planned_matmul_key(s, bt);
  if (key == nullptr) {
    PyErr_Clear();
    return nullptr;
  }
  SpmmCsrPlan* plan = planned_matmul_find(plans, key);
  Py_DECREF(key);
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
  c10::optional<torch::Tensor> served;
  try {
    served = plan->run(THPVariable_Unpack(values), bt, planned_matmul_nthreads(s), s.atparallel);
  } catch (...) {
    Py_DECREF(values);
    throw;  // translated to the Python exception by the caller's HANDLE_TH_ERRORS
  }
  Py_DECREF(values);
  if (!served.has_value()) return nullptr;
  return THPVariable_Wrap(std::move(*served));
}

// ---------------------------------------------------------------------------
// Torch sparse CSR operands
//
// Whether `a` is one this path may serve: exactly torch.Tensor, as the Python probe requires,
// and two-dimensional, as every plan is.
inline bool planned_matmul_torch_csr(const PlannedMatmul& s, PyObject* a) {
  if (reinterpret_cast<PyObject*>(Py_TYPE(a)) != s.tensor_type) return false;
  const at::Tensor& t = THPVariable_Unpack(a);
  return t.layout() == c10::kSparseCsr && t.dim() == 2;
}

// The state dict of `a`'s entry in plan.TORCH_OPERANDS (borrowed), or nullptr with no error set
// when it has none -- including an entry left by an earlier object with the same id, whose weak
// reference no longer names `a`.
inline PyObject* planned_matmul_torch_state(const PlannedMatmul& s, PyObject* a, PyObject* id) {
  if (s.torch_operands == nullptr || !PyDict_CheckExact(s.torch_operands)) return nullptr;
  PyObject* held = PyDict_GetItemWithError(s.torch_operands, id);  // borrowed
  if (held == nullptr || !PyTuple_CheckExact(held) || PyTuple_GET_SIZE(held) != 2) {
    PyErr_Clear();
    return nullptr;
  }
  PyObject* ref = PyTuple_GET_ITEM(held, 0);
  if (!PyWeakref_Check(ref) || PyWeakref_GetObject(ref) != a) return nullptr;
  PyObject* state = PyTuple_GET_ITEM(held, 1);
  return PyDict_CheckExact(state) ? state : nullptr;
}

// Records the first sighting of the product `key` for `a`: plan.install's rule that a plan is
// built only for an operand seen before, applied here so that an operand made afresh for every
// call -- one that is only ever sighted, never planned -- pays for it in C rather than in a
// cold Python function at the end of its ordinary path (2 us of a mkt1 call). Never fails the
// call it follows: any error is cleared.
inline void planned_matmul_record_sighting(const PlannedMatmul& s, PyObject* a, PyObject* id,
                                           PyObject* key) {
  PyObject* state = planned_matmul_torch_state(s, a, id);
  if (state != nullptr) {
    PyObject* seen = PyDict_GetItemWithError(state, s.seen_attr);  // borrowed
    if (seen != nullptr) {
      if (PySet_Check(seen) && PySet_GET_SIZE(seen) < 2 * s.max_plans && PySet_Add(seen, key) < 0)
        PyErr_Clear();
      return;
    }
    if (PyErr_Occurred()) {
      PyErr_Clear();
      return;
    }
    PyObject* fresh = PySet_New(nullptr);
    if (fresh == nullptr || PySet_Add(fresh, key) < 0 ||
        PyDict_SetItem(state, s.seen_attr, fresh) < 0)
      PyErr_Clear();
    Py_XDECREF(fresh);
    return;
  }
  // No entry yet: (weak reference carrying its key, {SEEN_ATTR: {key}}), as
  // plan.torch_operand_state(a, create=True) would make it.
  PyObject* ref = PyObject_CallFunctionObjArgs(s.operand_ref_type, a, s.operand_died, nullptr);
  if (ref == nullptr) {
    PyErr_Clear();
    return;
  }
  PyObject* seen = PySet_New(nullptr);
  PyObject* fresh_state = PyDict_New();
  PyObject* entry = nullptr;
  bool ok = seen != nullptr && fresh_state != nullptr &&
            PyObject_SetAttr(ref, s.key_attr, id) == 0 && PySet_Add(seen, key) == 0 &&
            PyDict_SetItem(fresh_state, s.seen_attr, seen) == 0;
  if (ok) {
    entry = PyTuple_Pack(2, ref, fresh_state);
    ok = entry != nullptr && PyDict_SetItem(s.torch_operands, id, entry) == 0;
  }
  if (!ok) PyErr_Clear();
  Py_XDECREF(entry);
  Py_XDECREF(fresh_state);
  Py_XDECREF(seen);
  Py_DECREF(ref);
}

// What the ordinary path has to do about plans for a torch operand's call this path did not
// serve -- plan.install's early exits, decided here so that only a call that has work for it
// reaches it.
enum class TorchOperandPlanWork {
  kNone,          // a withdrawn product, or a full cache: plan.install would return at once
  kFirstSighting,  // record the sighting here once the ordinary path has returned
  kInstall,       // seen before, or a plan declined: plan.install builds or accounts for it
};

// A torch sparse CSR operand: serve it from its plan, or hand it to the ordinary path under the
// probed keyword, whose value is the key when plan.install has work to do and None otherwise,
// so the Python function neither probes again nor calls the installer for nothing. A call
// whose ordinary path raises records nothing.
inline PyObject* planned_matmul_torch_call(const PlannedMatmul& s, PyObject* a, PyObject* b,
                                           PyObject* const* args, Py_ssize_t nargsf) {
  const at::Tensor& bt = THPVariable_Unpack(b);
  py::object key = py::reinterpret_steal<py::object>(planned_matmul_key(s, bt));
  py::object id = py::reinterpret_steal<py::object>(key ? PyLong_FromVoidPtr(a) : nullptr);
  if (!key || !id) {
    PyErr_Clear();
    return PyObject_Vectorcall(s.ordinary, args, nargsf, nullptr);
  }
  TorchOperandPlanWork work = TorchOperandPlanWork::kFirstSighting;
  if (PyObject* state = planned_matmul_torch_state(s, a, id.ptr())) {
    PyObject* plans = PyDict_GetItemWithError(state, s.plans_attr);  // borrowed
    PyObject* plan_obj = nullptr;
    if (plans != nullptr && PyDict_CheckExact(plans)) {
      plan_obj = PyDict_GetItemWithError(plans, key.ptr());  // borrowed
      if (plan_obj == Py_None) {
        work = TorchOperandPlanWork::kNone;
      } else if (plan_obj != nullptr) {
        work = TorchOperandPlanWork::kInstall;
        if (SpmmCsrPlan* plan = planned_matmul_cast(plan_obj)) {
          // The members themselves, not aliases: nothing is allocated, and a member shares
          // its version counter with every alias the caller may have written through.
          auto* impl = at::sparse_csr::get_sparse_csr_impl(THPVariable_Unpack(a));
          c10::optional<torch::Tensor> served =
              plan->run_csr(impl->compressed_indices(), impl->plain_indices(), impl->values(),
                            bt, planned_matmul_nthreads(s), s.atparallel);
          if (served.has_value()) return THPVariable_Wrap(std::move(*served));
        }
      } else if (PyDict_GET_SIZE(plans) >= s.max_plans) {
        work = TorchOperandPlanWork::kNone;
      }
    }
    PyErr_Clear();
    if (work == TorchOperandPlanWork::kFirstSighting) {
      PyObject* seen = PyDict_GetItemWithError(state, s.seen_attr);  // borrowed
      if (seen != nullptr && PySet_Check(seen) && PySet_Contains(seen, key.ptr()) == 1)
        work = TorchOperandPlanWork::kInstall;
      PyErr_Clear();
    }
  }
  PyObject* argv[3] = {a, b, work == TorchOperandPlanWork::kInstall ? key.ptr() : Py_None};
  PyObject* result = PyObject_Vectorcall(s.ordinary, argv, 2, s.probed_kwnames);
  if (result != nullptr && work == TorchOperandPlanWork::kFirstSighting)
    planned_matmul_record_sighting(s, a, id.ptr(), key.ptr());
  return result;
}

inline PyObject* planned_matmul_call(PyObject* /*module*/, PyObject* const* args,
                                     Py_ssize_t nargsf, PyObject* kwnames) {
  HANDLE_TH_ERRORS
  const PlannedMatmul& s = planned_matmul_state();
  if (PyVectorcall_NARGS(nargsf) == 2 && kwnames == nullptr &&
      reinterpret_cast<PyObject*>(Py_TYPE(args[1])) == s.tensor_type &&
      PyList_CheckExact(s.enabled) && PyList_GET_SIZE(s.enabled) >= 1 &&
      PyList_GET_ITEM(s.enabled, 0) == Py_True) {
    if (reinterpret_cast<PyObject*>(Py_TYPE(args[0])) == s.stensor_type) {
      PyObject* served = planned_matmul_try(s, args[0], args[1]);
      if (served != nullptr || PyErr_Occurred()) return served;
    } else if (planned_matmul_torch_csr(s, args[0])) {
      return planned_matmul_torch_call(s, args[0], args[1], args, nargsf);
    }
  }
  return PyObject_Vectorcall(s.ordinary, args, nargsf, kwnames);
  END_HANDLE_TH_ERRORS
}

inline void planned_matmul_set(PyObject*& slot, PyObject* value) {
  Py_XINCREF(value);
  Py_XSETREF(slot, value);
}

inline py::object install_planned_matmul(py::object ordinary, py::object stensor_type,
                                         py::object tensor_type, py::object generation,
                                         py::object enabled, const std::string& plans_attr,
                                         py::object torch_operands, py::object operand_ref_type,
                                         py::object operand_died, const std::string& seen_attr,
                                         Py_ssize_t max_plans, const std::string& probed_kw,
                                         bool match_host_threads, bool atparallel,
                                         py::object module, const std::string& doc) {
  if (!PyCallable_Check(ordinary.ptr())) throw py::type_error("ordinary must be callable");
  if (!PyType_Check(stensor_type.ptr()) || !PyType_Check(tensor_type.ptr()) ||
      !PyType_Check(operand_ref_type.ptr()))
    throw py::type_error("stensor_type, tensor_type and operand_ref_type must be types");
  if (!PyList_CheckExact(generation.ptr()) || PyList_GET_SIZE(generation.ptr()) < 1 ||
      !PyList_CheckExact(enabled.ptr()) || PyList_GET_SIZE(enabled.ptr()) < 1)
    throw py::type_error("generation and enabled must be one-element lists");
  if (!PyDict_CheckExact(torch_operands.ptr()))
    throw py::type_error("torch_operands must be a dict");
  if (!PyCallable_Check(operand_died.ptr())) throw py::type_error("operand_died must be callable");
  PlannedMatmul& s = planned_matmul_state();
  planned_matmul_set(s.ordinary, ordinary.ptr());
  planned_matmul_set(s.stensor_type, stensor_type.ptr());
  planned_matmul_set(s.tensor_type, tensor_type.ptr());
  planned_matmul_set(s.generation, generation.ptr());
  planned_matmul_set(s.enabled, enabled.ptr());
  planned_matmul_set(s.torch_operands, torch_operands.ptr());
  planned_matmul_set(s.operand_ref_type, operand_ref_type.ptr());
  planned_matmul_set(s.operand_died, operand_died.ptr());
  Py_XSETREF(s.plans_attr, PyUnicode_InternFromString(plans_attr.c_str()));
  Py_XSETREF(s.seen_attr, PyUnicode_InternFromString(seen_attr.c_str()));
  Py_XSETREF(s.probed_kwnames, Py_BuildValue("(s)", probed_kw.c_str()));
  if (s.storage_attr == nullptr) s.storage_attr = PyUnicode_InternFromString("_storage");
  if (s.value_attr == nullptr) s.value_attr = PyUnicode_InternFromString("_value");
  if (s.key_attr == nullptr) s.key_attr = PyUnicode_InternFromString("key");
  if (s.plans_attr == nullptr || s.seen_attr == nullptr || s.probed_kwnames == nullptr ||
      s.storage_attr == nullptr || s.value_attr == nullptr || s.key_attr == nullptr)
    throw py::error_already_set();
  s.max_plans = max_plans;
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
