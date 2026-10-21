// The memo key of a schedule's stored state (scorch.compiler.scheduler, "_schedule_state_key"),
// in a translation unit of its own: it needs only the CPython API, not torch.
//
// Why this is C. Schedule.cache_key validates the schedule's stored state on every read, and a
// memo keyed by that state's value spares the validation when the state was seen before. The key
// may be built only from a state the validation would read the same way, and without running any
// caller code, so building it checks the carrier's class, its stored names, each stored value's
// type and the class's entries at the field names. In Python each of those checks is a sweep over
// the fields, and a sweep costs about 1,540 instructions plus about 200 per field in CPython 3.12
// (sapling 85683): the Python key alone costs more than 864745d's whole read. Here each check is a
// pointer comparison.
//
// _schedule_state_key(value, table) -> tuple or None. table is (root, nested): root describes
// Schedule, nested the carriers a schedule holds (TileSpec, RelayoutSpec). Each description is
// (cls, mro, fields, allowed, items): the class; its method resolution order when the scheduler
// loaded, which must still be the class's own tuple; the stored names in declaration order; for
// each field, the exact types it admits; and for each field, None, str (each item of the tuple is
// an exact str) or a nested carrier class (each item is replaced by its own key). The scheduler
// builds the table once, from the tables its Python version reads.
//
// None means the memo does not key this state: the caller then validates it in full. The answer
// is None wherever the Python version's is (scheduler.py, _stored_values and the key functions):
//   - the value is not exactly the class, or the class's method resolution order is not the one
//     recorded, or the class's own dict defines __getattribute__ or __getattr__;
//   - reading the instance dict (PyObject_GenericGetDict, the getter of the class's __dict__
//     descriptor, which object.__getattribute__(value, "__dict__") reaches) raises an Exception;
//   - the stored state is not an exact dict whose keys are the field names themselves (compared
//     by pointer), in declaration order;
//   - the class's own dict holds a data descriptor at a field name. With the order (cls, object)
//     and no hook, the generic read of a field then returns the stored object;
//   - a stored value is not of an exact type its position admits, or an item of a tuple is not.
// Otherwise the answer is a new tuple of the stored objects themselves, in field order, each
// nested carrier replaced by its key.
//
// No caller code runs. No stored value is hashed, compared or has an attribute read, and no
// descriptor's __get__ runs. Types are compared by pointer. The class-dict lookups hash the field
// names, interned exact str whose hash is cached. The stored values are copied out of the dict in
// one walk that allocates nothing, so a collection (and a finalizer it runs) cannot change the dict
// mid-walk.
//
// Public CPython API only (no _Py names): the class dict is PyType_GetDict's on 3.12 and later,
// tp_dict on 3.11; the instance dict is PyObject_GenericGetDict's. Not for the free-threaded
// build.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

namespace {

// Interned at registration; live as long as the process.
PyObject* name_getattribute = nullptr;
PyObject* name_getattr = nullptr;

constexpr Py_ssize_t kDescriptionLength = 5;

// One carrier's description, borrowed from the table.
struct Description {
  PyTypeObject* cls;
  PyObject* mro;
  PyObject* fields;
  PyObject* allowed;
  PyObject* items;
  Py_ssize_t count;
};

bool malformed(const char* what) {
  PyErr_Format(PyExc_TypeError, "_schedule_state_key: malformed table: %s", what);
  return false;
}

// Reads a description; false, with TypeError set, if it is malformed.
bool read_description(PyObject* object, Description& out) {
  if (!PyTuple_CheckExact(object) || PyTuple_GET_SIZE(object) != kDescriptionLength) {
    return malformed("a description is not a tuple of five");
  }
  PyObject* cls = PyTuple_GET_ITEM(object, 0);
  out.mro = PyTuple_GET_ITEM(object, 1);
  out.fields = PyTuple_GET_ITEM(object, 2);
  out.allowed = PyTuple_GET_ITEM(object, 3);
  out.items = PyTuple_GET_ITEM(object, 4);
  if (!PyType_Check(cls) || !PyTuple_CheckExact(out.mro) || !PyTuple_CheckExact(out.fields) ||
      !PyTuple_CheckExact(out.allowed) || !PyTuple_CheckExact(out.items)) {
    return malformed("a description's parts are not a class and four tuples");
  }
  out.cls = reinterpret_cast<PyTypeObject*>(cls);
  out.count = PyTuple_GET_SIZE(out.fields);
  if (PyTuple_GET_SIZE(out.allowed) != out.count || PyTuple_GET_SIZE(out.items) != out.count) {
    return malformed("a description's per-field tuples differ in length");
  }
  for (Py_ssize_t i = 0; i < out.count; ++i) {
    PyObject* item = PyTuple_GET_ITEM(out.items, i);
    if (!PyUnicode_CheckExact(PyTuple_GET_ITEM(out.fields, i)) ||
        !PyTuple_CheckExact(PyTuple_GET_ITEM(out.allowed, i)) ||
        (item != Py_None && !PyType_Check(item))) {
      return malformed("a field's name, types or items rule has the wrong type");
    }
  }
  return true;
}

// The class's own dict, as a new reference; nullptr (with an error set) if it has none.
PyObject* class_dict(PyTypeObject* cls) {
#if PY_VERSION_HEX >= 0x030C0000
  return PyType_GetDict(cls);
#else
  PyObject* dict = cls->tp_dict;
  if (dict == nullptr) {
    PyErr_SetString(PyExc_TypeError, "_schedule_state_key: a class without a dict");
    return nullptr;
  }
  Py_INCREF(dict);
  return dict;
#endif
}

PyObject* refuse() { Py_RETURN_NONE; }

// 1 if the class dict defines neither attribute hook, 0 if it defines one, -1 on error.
int no_hooks(PyObject* dict) {
  int has = PyDict_Contains(dict, name_getattribute);
  if (has == 0) has = PyDict_Contains(dict, name_getattr);
  return has < 0 ? -1 : has == 0;
}

// 1 if the class dict holds no data descriptor at any field name, 0 if it holds one, -1 on error.
int no_data_descriptor(PyObject* dict, const Description& d) {
  for (Py_ssize_t i = 0; i < d.count; ++i) {
    PyObject* entry = PyDict_GetItemWithError(dict, PyTuple_GET_ITEM(d.fields, i));
    if (entry == nullptr) {
      if (PyErr_Occurred()) return -1;
      continue;
    }
    if (Py_TYPE(entry)->tp_descr_set != nullptr) return 0;
  }
  return 1;
}

// The stored values in field order (a new tuple), None if refused, nullptr on error.
PyObject* stored_values(PyObject* value, const Description& d) {
  if (Py_TYPE(value) != d.cls || d.cls->tp_mro != d.mro) return refuse();
  PyObject* dict = class_dict(d.cls);
  if (dict == nullptr) return nullptr;
  const int hooks_absent = no_hooks(dict);
  if (hooks_absent <= 0) {
    Py_DECREF(dict);
    return hooks_absent < 0 ? nullptr : refuse();
  }
  // The instance dict, read without a type lookup. object.__getattribute__(value, "__dict__")
  // returns the same object: it finds the class's __dict__ descriptor, whose getter is this
  // function, and type.__setattr__ cannot replace that descriptor.
  PyObject* state = PyObject_GenericGetDict(value, nullptr);
  if (state == nullptr) {
    Py_DECREF(dict);
    if (!PyErr_ExceptionMatches(PyExc_Exception)) return nullptr;
    PyErr_Clear();
    return refuse();
  }
  if (!PyDict_CheckExact(state) || PyDict_GET_SIZE(state) != d.count) {
    Py_DECREF(state);
    Py_DECREF(dict);
    return refuse();
  }
  // Allocated before the walk: nothing inside the walk allocates.
  PyObject* values = PyTuple_New(d.count);
  if (values == nullptr) {
    Py_DECREF(state);
    Py_DECREF(dict);
    return nullptr;
  }
  Py_ssize_t position = 0;
  Py_ssize_t index = 0;
  PyObject* key = nullptr;
  PyObject* stored = nullptr;
  bool same_names = true;
  while (PyDict_Next(state, &position, &key, &stored)) {
    if (index >= d.count || key != PyTuple_GET_ITEM(d.fields, index)) {
      same_names = false;
      break;
    }
    Py_INCREF(stored);
    PyTuple_SET_ITEM(values, index, stored);
    ++index;
  }
  Py_DECREF(state);
  if (!same_names || index != d.count) {
    Py_DECREF(values);
    Py_DECREF(dict);
    return refuse();
  }
  const int plain = no_data_descriptor(dict, d);
  Py_DECREF(dict);
  if (plain <= 0) {
    Py_DECREF(values);
    return plain < 0 ? nullptr : refuse();
  }
  return values;
}

bool admitted(PyObject* value, PyObject* types) {
  const Py_ssize_t n = PyTuple_GET_SIZE(types);
  for (Py_ssize_t i = 0; i < n; ++i) {
    if (reinterpret_cast<PyObject*>(Py_TYPE(value)) == PyTuple_GET_ITEM(types, i)) return true;
  }
  return false;
}

// Looks up the nested description whose class is cls: true with present set when there is one,
// true with present unset when there is none, false (TypeError set) if the table is malformed.
bool find_nested(PyObject* nested, PyObject* cls, Description& found, bool& present) {
  present = false;
  const Py_ssize_t n = PyTuple_GET_SIZE(nested);
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* description = PyTuple_GET_ITEM(nested, i);
    if (!PyTuple_CheckExact(description) || PyTuple_GET_SIZE(description) != kDescriptionLength) {
      return malformed("a nested description is not a tuple of five");
    }
    if (PyTuple_GET_ITEM(description, 0) == cls) {
      if (!read_description(description, found)) return false;
      present = true;
      return true;
    }
  }
  return true;
}

PyObject* state_key(PyObject* value, const Description& d, PyObject* nested, bool root);

// The key that replaces a nested carrier: a new reference, None if refused, nullptr on error.
PyObject* nested_key(PyObject* value, PyObject* nested, PyObject* cls) {
  Description inner;
  bool present = false;
  if (!find_nested(nested, cls, inner, present)) return nullptr;
  if (!present) {
    malformed("an items rule names a class with no nested description");
    return nullptr;
  }
  return state_key(value, inner, nested, false);
}

// Replaces values[i] (owned, not yet shared) with replacement (a new reference).
void replace(PyObject* values, Py_ssize_t i, PyObject* replacement) {
  PyObject* old = PyTuple_GET_ITEM(values, i);
  PyTuple_SET_ITEM(values, i, replacement);
  Py_DECREF(old);
}

PyObject* state_key(PyObject* value, const Description& d, PyObject* nested, bool root) {
  PyObject* values = stored_values(value, d);
  if (values == nullptr || values == Py_None) return values;
  for (Py_ssize_t i = 0; i < d.count; ++i) {
    PyObject* stored = PyTuple_GET_ITEM(values, i);
    if (!admitted(stored, PyTuple_GET_ITEM(d.allowed, i))) {
      Py_DECREF(values);
      return refuse();
    }
    PyObject* rule = PyTuple_GET_ITEM(d.items, i);
    if (rule == Py_None) {
      if (stored == Py_None) continue;
      Description inner;
      bool present = false;
      if (!find_nested(nested, reinterpret_cast<PyObject*>(Py_TYPE(stored)), inner, present)) {
        Py_DECREF(values);
        return nullptr;
      }
      if (!present) continue;
      if (!root) {
        Py_DECREF(values);
        malformed("a nested carrier admits another carrier");
        return nullptr;
      }
      PyObject* key = state_key(stored, inner, nested, false);
      if (key == nullptr || key == Py_None) {
        Py_DECREF(values);
        return key;
      }
      replace(values, i, key);
    } else if (rule == reinterpret_cast<PyObject*>(&PyUnicode_Type)) {
      if (!PyTuple_CheckExact(stored)) continue;
      const Py_ssize_t n = PyTuple_GET_SIZE(stored);
      for (Py_ssize_t j = 0; j < n; ++j) {
        if (!PyUnicode_CheckExact(PyTuple_GET_ITEM(stored, j))) {
          Py_DECREF(values);
          return refuse();
        }
      }
    } else {
      if (!root) {
        Py_DECREF(values);
        malformed("a nested carrier's items rule names a carrier");
        return nullptr;
      }
      if (!PyTuple_CheckExact(stored) || PyTuple_GET_SIZE(stored) == 0) continue;
      const Py_ssize_t n = PyTuple_GET_SIZE(stored);
      PyObject* keys = PyTuple_New(n);
      if (keys == nullptr) {
        Py_DECREF(values);
        return nullptr;
      }
      for (Py_ssize_t j = 0; j < n; ++j) {
        PyObject* key = nested_key(PyTuple_GET_ITEM(stored, j), nested, rule);
        if (key == nullptr || key == Py_None) {
          Py_DECREF(keys);
          Py_DECREF(values);
          return key;
        }
        PyTuple_SET_ITEM(keys, j, key);
      }
      replace(values, i, keys);
    }
  }
  return values;
}

// _schedule_state_key(value, table) -> tuple or None.
PyObject* schedule_state_key(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
  if (nargs != 2) {
    PyErr_SetString(PyExc_TypeError, "_schedule_state_key takes a value and a table");
    return nullptr;
  }
  PyObject* table = args[1];
  if (!PyTuple_CheckExact(table) || PyTuple_GET_SIZE(table) != 2 ||
      !PyTuple_CheckExact(PyTuple_GET_ITEM(table, 1))) {
    malformed("the table is not (root, nested)");
    return nullptr;
  }
  Description root;
  if (!read_description(PyTuple_GET_ITEM(table, 0), root)) return nullptr;
  return state_key(args[0], root, PyTuple_GET_ITEM(table, 1), true);
}

PyMethodDef module_functions[] = {
    {"_schedule_state_key",
     reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(schedule_state_key)),
     METH_FASTCALL,
     "_schedule_state_key(value, table): the memo key of a schedule's stored state, or None"},
    {nullptr, nullptr, 0, nullptr},
};

}  // namespace

namespace scorch {

// Called from scorch_ops' module init (ops.cpp).
int register_schedule_key(PyObject* module) {
  name_getattribute = PyUnicode_InternFromString("__getattribute__");
  name_getattr = PyUnicode_InternFromString("__getattr__");
  if (name_getattribute == nullptr || name_getattr == nullptr) return -1;
  return PyModule_AddFunctions(module, module_functions);
}

}  // namespace scorch
