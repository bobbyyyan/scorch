// Stored state for Scorch's value types (scorch.format.LevelFormat and TensorFormat,
// scorch.layout.TensorLayout and TensorMetadata), in a translation unit of its own: it needs
// only the CPython API, not torch.
//
// Why this is C. Those four classes are frozen dataclasses, and a frozen dataclass is frozen
// only by convention: object.__setattr__ writes its fields. So a value object could never be
// shared -- a cache that handed one layout to two tensors let a write through one tensor's
// layout rewrite the other's -- and every boundary that retained one had to audit and copy
// it. Python has no read-only instance field. A slot's descriptor can be removed from the
// class, but a class with the same slot names is layout-compatible, so assigning __class__
// to it reaches the slot through that class's descriptor. A C type can declare its fields
// READONLY, which object.__setattr__ and object.__delattr__ both refuse.
//
// Each value class subclasses a base made here (sealed_type) that holds its fields. A field
// starts unset and is set exactly once, by seal(), which the Python constructor calls after
// validating; seal() refuses an object whose fields are already set, so no call -- not even
// one through these private functions -- changes an object once it exists. A base admits one
// direct subclass, the value class, and that class's own subclasses must add state (a
// __dict__ or slots), so no class is layout-compatible with a value class: assigning
// __class__ on one of its instances fails. Each base is an immutable type, so its own
// attributes cannot be replaced either.
//
// The bases do not take part in cyclic garbage collection. Their fields hold only values a
// constructor validated -- tuples of ints, enum members, dtypes, strings and other value
// objects, each of which existed before the object holding it -- so no reference cycle can
// pass through a field. A subclass that adds a __dict__ is collected through that dict by
// the interpreter's own subclass support.

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include <cstddef>
#include <cstring>
#include <vector>

namespace {

struct SealedBase {
  PyTypeObject* type;      // the base sealed_type made
  Py_ssize_t nfields;
  PyTypeObject* designated;  // its one direct subclass, once defined
};

// Every base made, in creation order. A handful per process (one per value class), so a
// linear search is the fastest lookup.
std::vector<SealedBase>& bases() {
  static std::vector<SealedBase> all;
  return all;
}

constexpr Py_ssize_t kWeaklistOffset = sizeof(PyObject);
constexpr Py_ssize_t kFieldsOffset = sizeof(PyObject) + sizeof(PyObject*);

inline PyObject** fields_of(PyObject* self) {
  return reinterpret_cast<PyObject**>(reinterpret_cast<char*>(self) + kFieldsOffset);
}

// The base in type's chain of tp_base, or nullptr. A value class's instances are two steps
// from their base; a subclass's are three.
SealedBase* base_of(PyTypeObject* type) {
  for (PyTypeObject* t = type; t != nullptr; t = t->tp_base) {
    for (SealedBase& entry : bases()) {
      if (entry.type == t) return &entry;
    }
  }
  return nullptr;
}

// A base inherits object's allocator, so a value class's instances start with every field
// unset and its own __init__ validates the arguments. Pickles written with protocol 0 or 1
// rebuild an object through object.__new__(cls), which CPython refuses for a class whose
// allocator is not object's. This __init__ only refuses instances of a base itself, which
// a value class's __init__ overrides; a base's instances never hold fields (seal_into).
int sealed_init(PyObject* self, PyObject*, PyObject*) {
  PyTypeObject* type = Py_TYPE(self);
  SealedBase* base = base_of(type);
  if (base == nullptr || base->type == type) {
    PyErr_Format(PyExc_TypeError, "cannot create '%s' instances", type->tp_name);
    return -1;
  }
  return 0;
}

void sealed_dealloc(PyObject* self) {
  PyTypeObject* type = Py_TYPE(self);
  if (*reinterpret_cast<PyObject**>(reinterpret_cast<char*>(self) + kWeaklistOffset)) {
    PyObject_ClearWeakRefs(self);
  }
  SealedBase* base = base_of(type);
  if (base != nullptr) {
    PyObject** fields = fields_of(self);
    for (Py_ssize_t i = 0; i < base->nfields; ++i) Py_CLEAR(fields[i]);
  }
  type->tp_free(self);
  // An instance of a heap type holds a reference to its type.
  Py_DECREF(type);
}

// The base's __init_subclass__. The first direct subclass becomes the value class; a second
// direct subclass is refused, and so is any later subclass that adds no state, because
// either would be layout-compatible with the value class.
PyObject* sealed_init_subclass(PyObject* cls, PyObject* args, PyObject* kwargs) {
  // object.__init_subclass__ takes no arguments, and this replaces it.
  if (PyTuple_GET_SIZE(args) != 0 || (kwargs != nullptr && PyDict_GET_SIZE(kwargs) != 0)) {
    PyErr_SetString(PyExc_TypeError, "__init_subclass__() takes no arguments");
    return nullptr;
  }
  PyTypeObject* type = reinterpret_cast<PyTypeObject*>(cls);
  SealedBase* base = base_of(type);
  if (base == nullptr) Py_RETURN_NONE;
  if (type->tp_base == base->type) {
    if (base->designated != nullptr) {
      PyErr_Format(PyExc_TypeError,
                   "'%s' holds the fields of '%s' and cannot be subclassed again",
                   base->type->tp_name, base->designated->tp_name);
      return nullptr;
    }
    // The value class itself holds nothing but the base's fields: no __dict__, no slots.
    if (type->tp_basicsize != base->type->tp_basicsize || type->tp_dictoffset != 0 ||
        type->tp_itemsize != 0) {
      PyErr_Format(PyExc_TypeError,
                   "'%s' must declare __slots__ = (): its fields are those of '%s'",
                   type->tp_name, base->type->tp_name);
      return nullptr;
    }
    Py_INCREF(type);
    base->designated = type;
    Py_RETURN_NONE;
  }
  PyTypeObject* designated = base->designated;
  if (designated != nullptr && type->tp_basicsize == designated->tp_basicsize &&
      type->tp_itemsize == designated->tp_itemsize &&
      type->tp_dictoffset == designated->tp_dictoffset &&
      type->tp_weaklistoffset == designated->tp_weaklistoffset) {
    PyErr_Format(PyExc_TypeError,
                 "a subclass of '%s' must add state (a __dict__ or slots): '%s' adds none, "
                 "so it would be layout-compatible with '%s' and assigning __class__ could "
                 "turn an existing value into one",
                 designated->tp_name, type->tp_name, designated->tp_name);
    return nullptr;
  }
  Py_RETURN_NONE;
}

PyMethodDef sealed_methods[] = {
    {"__init_subclass__", reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(
                              sealed_init_subclass)),
     METH_VARARGS | METH_KEYWORDS | METH_CLASS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

// sealed_type(name, fields) -> a new base whose instances hold len(fields) read-only fields
// under those names. The member table and its names live as long as the type, which is the
// life of the process.
PyObject* sealed_type(PyObject*, PyObject* args) {
  const char* name = nullptr;
  PyObject* names = nullptr;
  if (!PyArg_ParseTuple(args, "sO!:_sealed_type", &name, &PyTuple_Type, &names)) {
    return nullptr;
  }
  const Py_ssize_t n = PyTuple_GET_SIZE(names);
  if (n < 1) {
    PyErr_SetString(PyExc_ValueError, "_sealed_type needs at least one field");
    return nullptr;
  }
  auto* members = new PyMemberDef[n + 2]();
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* field = PyTuple_GET_ITEM(names, i);
    const char* text = PyUnicode_Check(field) ? PyUnicode_AsUTF8(field) : nullptr;
    if (text == nullptr) {
      if (!PyErr_Occurred()) PyErr_SetString(PyExc_TypeError, "field names must be str");
      for (Py_ssize_t j = 0; j < i; ++j) delete[] const_cast<char*>(members[j].name);
      delete[] members;
      return nullptr;
    }
    const size_t length = std::strlen(text);
    char* copy = new char[length + 1];
    std::memcpy(copy, text, length + 1);
    members[i].name = copy;
    members[i].type = T_OBJECT_EX;  // an unset field raises AttributeError when read
    members[i].offset = kFieldsOffset + i * static_cast<Py_ssize_t>(sizeof(PyObject*));
    members[i].flags = READONLY;
  }
  members[n].name = "__weaklistoffset__";
  members[n].type = T_PYSSIZET;
  members[n].offset = kWeaklistOffset;
  members[n].flags = READONLY;

  PyType_Slot slots[] = {
      {Py_tp_init, reinterpret_cast<void*>(sealed_init)},
      {Py_tp_dealloc, reinterpret_cast<void*>(sealed_dealloc)},
      {Py_tp_members, members},
      {Py_tp_methods, sealed_methods},
      {0, nullptr},
  };
  // The qualified name, so the type reports module scorch_ops.
  const size_t length = std::strlen(name);
  char* qualified = new char[length + sizeof("scorch_ops.")];
  std::memcpy(qualified, "scorch_ops.", sizeof("scorch_ops.") - 1);
  std::memcpy(qualified + sizeof("scorch_ops.") - 1, name, length + 1);
  PyType_Spec spec = {
      qualified,
      static_cast<int>(kFieldsOffset + n * static_cast<Py_ssize_t>(sizeof(PyObject*))),
      0,
      Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_IMMUTABLETYPE,
      slots,
  };
  PyObject* type = PyType_FromSpec(&spec);
  if (type == nullptr) return nullptr;
  bases().push_back({reinterpret_cast<PyTypeObject*>(type), n, nullptr});
  return type;
}

// Set every field of an object whose fields are all unset. False if any is already set,
// which is how a second __init__ on an existing value is refused.
int seal_into(PyObject* self, PyObject* const* values, Py_ssize_t count) {
  SealedBase* base = base_of(Py_TYPE(self));
  if (base == nullptr || base->type == Py_TYPE(self)) {
    PyErr_SetString(PyExc_TypeError, "not a sealed value");
    return -1;
  }
  if (count != base->nfields) {
    PyErr_Format(PyExc_TypeError, "'%s' has %zd fields, got %zd values",
                 Py_TYPE(self)->tp_name, base->nfields, count);
    return -1;
  }
  PyObject** fields = fields_of(self);
  for (Py_ssize_t i = 0; i < count; ++i) {
    if (fields[i] != nullptr) return 0;
  }
  for (Py_ssize_t i = 0; i < count; ++i) {
    Py_INCREF(values[i]);
    fields[i] = values[i];
  }
  return 1;
}

// _seal(obj, *values) -> bool.
PyObject* seal(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
  if (nargs < 1) {
    PyErr_SetString(PyExc_TypeError, "_seal needs an object");
    return nullptr;
  }
  const int sealed = seal_into(args[0], args + 1, nargs - 1);
  if (sealed < 0) return nullptr;
  return PyBool_FromLong(sealed);
}

// _sealed_new(cls, *values) -> a new instance of cls with its fields set, without running
// cls.__init__: for values Scorch has already validated.
PyObject* sealed_new_from(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
  if (nargs < 1 || !PyType_Check(args[0])) {
    PyErr_SetString(PyExc_TypeError, "_sealed_new needs a class");
    return nullptr;
  }
  auto* type = reinterpret_cast<PyTypeObject*>(args[0]);
  SealedBase* base = base_of(type);
  if (base == nullptr || base->type == type) {
    PyErr_Format(PyExc_TypeError, "cannot create '%s' instances", type->tp_name);
    return nullptr;
  }
  PyObject* self = type->tp_alloc(type, 0);
  if (self == nullptr) return nullptr;
  if (seal_into(self, args + 1, nargs - 1) < 0) {
    Py_DECREF(self);
    return nullptr;
  }
  return self;
}

PyMethodDef module_functions[] = {
    {"_sealed_type", sealed_type, METH_VARARGS,
     "_sealed_type(name, fields): a base whose instances hold read-only fields, each set "
     "once by _seal"},
    {"_seal", reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(seal)),
     METH_FASTCALL,
     "_seal(obj, *values): set every field of an object none of whose fields is set; "
     "False if any already is"},
    {"_sealed_new", reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(sealed_new_from)),
     METH_FASTCALL, "_sealed_new(cls, *values): a new instance with its fields set"},
    {nullptr, nullptr, 0, nullptr},
};

}  // namespace

namespace scorch {

// Called from scorch_ops' module init (ops.cpp).
int register_sealed_values(PyObject* module) {
  return PyModule_AddFunctions(module, module_functions);
}

}  // namespace scorch
