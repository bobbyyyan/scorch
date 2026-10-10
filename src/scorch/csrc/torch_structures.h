#pragma once

// Plan state for torch sparse CSR operands, keyed on their index arrays.
//
// A torch CSR operand's plans, sightings, autograd verdict and x @ A records are facts about its
// STRUCTURE -- its two index arrays and its shape -- not about the tensor object: a plan takes the
// values on every call. Keyed on the object (plan.TORCH_OPERANDS, id(a) checked against a weak
// reference), a program that builds a new CSR tensor over the same arrays for every call --
// `torch.sparse_csr_tensor(crow, col, w)` with weights that change each step, as GAT and any
// learned edge weighting do -- made a first sighting on every call and was never planned: on
// mkt1's 200-matrix sample such calls were behind MKL in 1152 of 2400 runs back to back and 2184
// of 2400 cold, where the same product on one persistent object was behind in none (sc3).
//
// So the state lives here, one entry per structure, found from the operand's members (the arrays
// it was built from, which are the caller's own: native_abi.h, Write witnesses) by their storages,
// buffers, lengths and dtypes and the operand's shape. Every object over the same arrays finds the
// same entry; arrays that differ in any of these find a different one.
//
// An entry must live exactly as long as its arrays. Nothing in it may hold them: the plans in it
// borrow them (SpmmCsrPlan::release_sources, TransposedProductPlan), and autograd.py keeps only a
// verdict about them. What it holds instead is a weak reference to each storage, which keeps the
// storage's allocation -- and so its address -- from being reused while the entry exists: a key
// that matches a live tensor's members is therefore that tensor's structure, never a successor
// that happens to occupy the same addresses. When the arrays die the entry is dropped: by a weak
// reference on each storage's Python object, whose callback erases it (torch keeps one Python
// object per storage for the storage's whole life, the property autograd._TRANSPOSES relies on);
// or, for an entry that never got those -- a first sighting made by the C entry, which must cost
// what it did before (see torch_structure_sight) -- by a sweep once the registry has doubled.
//
// Every operand found here is recorded as a write witness of its arrays (native_abi.h): any
// object over the arrays may write them through its accessors, and every cache over them must see
// it. The last object recorded through an entry is remembered, so a persistent operand takes no
// lock on its later calls. (An entry added for a first sighting records no witness: nothing is
// read from it until a later lookup, which records one, and a witness recorded late is safe --
// one whose counter has already moved advances the epoch as it is recorded.)
//
// An entry describes its arrays as they were when it was given its record (TorchStructureRecord):
// when Python first keeps something in it -- a plan, or a transposed product's record. Once they
// have been written in place -- through the caller's arrays, or through any operand's accessors,
// which torch cannot tell from a write to its values -- everything in it describes arrays that no
// longer exist, and the next call it does not serve CLEARS it (torch_structure_revalidate): plans,
// sightings, declines, records. The rule that a plan is installed on a structure's second sighting
// then applies afresh. Without that, a plan that went stale stayed in the entry declining every
// call, and no tensor over those arrays was planned again; with it, a structure written once is
// planned again two calls later, and one written before every call (values updated in place
// through values(), say) is sighted on every call and never planned, which is what it cost when
// plans were kept per tensor object.
//
// Everything here runs holding the GIL: the C entry point and the Python bindings are its only
// callers, and the weak-reference callbacks run under it too.

#include "native_abi.h"
#include "plan.h"

#include <ATen/SparseCsrTensorUtils.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Storage.h>

#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace scorch_native {

struct TorchStructureKey {
  const void* positions_storage;
  const void* coordinates_storage;
  const void* positions_data;
  const void* coordinates_data;
  int64_t positions_numel;
  int64_t coordinates_numel;
  int64_t rows;
  int64_t cols;
  int32_t positions_dtype;
  int32_t coordinates_dtype;

  bool operator==(const TorchStructureKey& o) const {
    return positions_storage == o.positions_storage &&
           coordinates_storage == o.coordinates_storage &&
           positions_data == o.positions_data && coordinates_data == o.coordinates_data &&
           positions_numel == o.positions_numel && coordinates_numel == o.coordinates_numel &&
           rows == o.rows && cols == o.cols && positions_dtype == o.positions_dtype &&
           coordinates_dtype == o.coordinates_dtype;
  }
};

struct TorchStructureKeyHash {
  size_t operator()(const TorchStructureKey& k) const {
    size_t h = std::hash<const void*>()(k.positions_storage);
    auto mix = [&h](size_t v) { h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); };
    mix(std::hash<const void*>()(k.coordinates_storage));
    mix(std::hash<const void*>()(k.positions_data));
    mix(std::hash<const void*>()(k.coordinates_data));
    mix(std::hash<int64_t>()(k.coordinates_numel));
    mix(std::hash<int64_t>()(k.rows));
    mix(std::hash<int64_t>()(k.cols));
    return h;
  }
};

// The arrays as an entry describes them: by identity, by their own counters and by their storages'
// write-witness epochs, as a borrowing plan records its sources (plan.h).
struct TorchStructureRecord {
  BorrowedArray positions;
  BorrowedArray coordinates;
  PlanSourceWatch positions_watch;
  PlanSourceWatch coordinates_watch;

  TorchStructureRecord(const at::Tensor& p, const at::Tensor& c)
      : positions(p), coordinates(c), positions_watch(p), coordinates_watch(c) {}

  bool current(const at::Tensor& p, const at::Tensor& c) const {
    return positions.is(p) && coordinates.is(c) && positions_watch.current() &&
           coordinates_watch.current();
  }
};

// A product sighted once through the C entry point, as plain data: the right operand's extents
// and dtype and the plan generation, which is what plan.py's key ((b rows, b cols), b dtype,
// generation) is made of (planned_matmul.h).
struct TorchSighting {
  int64_t dim = 0;
  int64_t extent[2] = {0, 0};
  int32_t dtype = 0;
  int64_t generation = 0;

  bool operator==(const TorchSighting& o) const {
    return dim == o.dim && extent[0] == o.extent[0] && extent[1] == o.extent[1] &&
           dtype == o.dtype && generation == o.generation;
  }
};

struct TorchStructureEntry {
  c10::weak_intrusive_ptr<c10::StorageImpl> positions_alive{
      c10::intrusive_ptr<c10::StorageImpl>()};
  c10::weak_intrusive_ptr<c10::StorageImpl> coordinates_alive{
      c10::intrusive_ptr<c10::StorageImpl>()};
  // The dict plan.py and autograd.py keep this structure's state in; none until Python first asks
  // to keep something (torch_structure_create).
  py::object state;
  py::object watchers;  // empty, or the weak references whose callbacks drop this entry
  uint64_t id = 0;
  // The version counter of the last operand recorded as a witness through this entry. Held, so
  // that its address -- which is what identifies it -- cannot be reused by another counter while
  // it is compared against.
  c10::optional<c10::VariableVersion> last_witness;
  // What `state` describes; none until Python first keeps something in it (see the top).
  std::unique_ptr<TorchStructureRecord> record;
  // Products the C entry has sighted while there was no `state`. They become plan.py's SEEN set
  // when `state` is made. Kept as plain data rather than as that set because a first sighting is
  // what a call over new arrays makes on every call, and a dict and a set for each one are two
  // containers the garbage collector tracks for as long as the arrays live (199 first sightings
  // kept 450 more tracked objects that way, and fewer than 20 this way). The plan installed on a
  // second sighting is all a sighting is for, and most structures never have one.
  std::vector<TorchSighting> sighted;

  bool live() const { return !positions_alive.expired() && !coordinates_alive.expired(); }
};

struct TorchStructures {
  std::unordered_map<TorchStructureKey, TorchStructureEntry, TorchStructureKeyHash> entries;
  std::unordered_map<uint64_t, TorchStructureKey> by_id;  // for the callbacks
  uint64_t next_id = 1;
  size_t swept_at = 0;
  // The weak references of entries dropped by their own callback, released on the next insert:
  // a callback must not free the weak reference that is calling it.
  std::vector<py::object> graveyard;
  // plan.SEEN_ATTR, interned, from planned_matmul.h's installer: the key `sighted` is moved under.
  py::object seen_attr;
};

// Never destroyed: it holds Python objects, which must not be released after the interpreter.
inline TorchStructures& torch_structures() {
  static TorchStructures* registry = new TorchStructures();
  return *registry;
}

// The members of the 2-D sparse CSR tensor `a` and the key of its structure; false when they
// have no storage to key on.
inline bool torch_structure_key(const at::Tensor& a, const at::Tensor*& positions,
                                const at::Tensor*& coordinates, TorchStructureKey& key) {
  auto* impl = at::sparse_csr::get_sparse_csr_impl(a);
  positions = &impl->compressed_indices();
  coordinates = &impl->plain_indices();
  if (!positions->has_storage() || !coordinates->has_storage()) return false;
  key.positions_storage = positions->storage().unsafeGetStorageImpl();
  key.coordinates_storage = coordinates->storage().unsafeGetStorageImpl();
  key.positions_data = positions->data_ptr();
  key.coordinates_data = coordinates->data_ptr();
  key.positions_numel = positions->numel();
  key.coordinates_numel = coordinates->numel();
  key.rows = a.size(0);
  key.cols = a.size(1);
  key.positions_dtype = static_cast<int32_t>(positions->scalar_type());
  key.coordinates_dtype = static_cast<int32_t>(coordinates->scalar_type());
  return true;
}

// Records `a` as a write witness of its arrays, unless it is the operand last recorded through
// `entry` (then it already is one: a witness leaves the table only once nothing holds its
// counter, and `a` does). Nothing for an operand without a version counter (an inference tensor).
inline void torch_structure_witness(TorchStructureEntry& entry, const at::Tensor& a,
                                    const at::Tensor& positions,
                                    const at::Tensor& coordinates) {
  const c10::VariableVersion& counter = a.unsafeGetTensorImpl()->version_counter();
  if (!counter.enabled()) return;
  if (entry.last_witness && abi_counter_id(*entry.last_witness) == abi_counter_id(counter)) {
    return;
  }
  abi_watch_register(positions, a);
  abi_watch_register(coordinates, a);
  entry.last_witness = counter;
}

// Removes the entry `it` names, releasing its state only once the maps are consistent again
// (releasing it can free plans, and with them run arbitrary destructors).
inline void torch_structure_erase(
    std::unordered_map<TorchStructureKey, TorchStructureEntry, TorchStructureKeyHash>::iterator
        it) {
  TorchStructures& r = torch_structures();
  py::object state = std::move(it->second.state);
  if (it->second.watchers) r.graveyard.push_back(std::move(it->second.watchers));
  r.by_id.erase(it->second.id);
  r.entries.erase(it);
}

// The weak-reference callback: the storage of one of entry `self`'s arrays has died.
inline PyObject* torch_structure_died(PyObject* self, PyObject* /*weakref*/) {
  TorchStructures& r = torch_structures();
  const uint64_t id = PyLong_AsUnsignedLongLong(self);
  if (PyErr_Occurred()) {
    PyErr_Clear();
    Py_RETURN_NONE;
  }
  auto key = r.by_id.find(id);
  if (key != r.by_id.end()) {
    auto it = r.entries.find(key->second);
    if (it != r.entries.end() && it->second.id == id) {
      torch_structure_erase(it);
    } else {
      r.by_id.erase(key);
    }
  }
  Py_RETURN_NONE;
}

// Gives `entry` the weak references that drop it when either array's storage dies. False, with no
// error set, when they could not be made; the entry is then dropped by a sweep instead.
inline bool torch_structure_attach(TorchStructureEntry& entry, const at::Tensor& positions,
                                   const at::Tensor& coordinates) {
  if (entry.watchers) return true;
  static PyMethodDef def = {
      "_torch_structure_died",
      reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)(void)>(torch_structure_died)),
      METH_O, nullptr};
  py::object self = py::reinterpret_steal<py::object>(PyLong_FromUnsignedLongLong(entry.id));
  py::object callback =
      self ? py::reinterpret_steal<py::object>(PyCFunction_New(&def, self.ptr())) : py::object();
  py::object positions_storage =
      py::reinterpret_steal<py::object>(THPStorage_Wrap(positions.storage()));
  const bool shared = coordinates.storage().unsafeGetStorageImpl() ==
                      positions.storage().unsafeGetStorageImpl();
  py::object coordinates_storage =
      shared ? positions_storage
             : py::reinterpret_steal<py::object>(THPStorage_Wrap(coordinates.storage()));
  if (!callback || !positions_storage || !coordinates_storage) {
    PyErr_Clear();
    return false;
  }
  py::object first = py::reinterpret_steal<py::object>(
      PyWeakref_NewRef(positions_storage.ptr(), callback.ptr()));
  py::object second =
      shared ? py::none()
             : py::reinterpret_steal<py::object>(
                   PyWeakref_NewRef(coordinates_storage.ptr(), callback.ptr()));
  if (!first || !second) {
    PyErr_Clear();
    return false;
  }
  entry.watchers = py::make_tuple(first, second);
  return true;
}

// `a`'s structure's entry, with `a`'s members, or nullptr when it has none. `a` must be a 2-D
// sparse CSR tensor.
struct TorchStructureFound {
  TorchStructureEntry* entry = nullptr;
  const at::Tensor* positions = nullptr;
  const at::Tensor* coordinates = nullptr;
};

inline TorchStructureEntry* torch_structure_find(const at::Tensor& a, TorchStructureFound& found,
                                                 TorchStructureKey& key) {
  found.entry = nullptr;
  if (!torch_structure_key(a, found.positions, found.coordinates, key)) return nullptr;
  TorchStructures& r = torch_structures();
  auto it = r.entries.find(key);
  if (it == r.entries.end()) return nullptr;
  // Live, necessarily: `a` holds storages at the key's addresses, and no storage can have been
  // made at the address of one an entry still references weakly, so this entry's are `a`'s.
  found.entry = &it->second;
  return found.entry;
}

inline TorchStructureEntry* torch_structure_find(const at::Tensor& a, TorchStructureFound& found) {
  TorchStructureKey key;
  return torch_structure_find(a, found, key);
}

// The state dict of `a`'s structure (borrowed), recording `a` as a witness, or nullptr with no
// error set when it has none.
inline PyObject* torch_structure_lookup(const at::Tensor& a) {
  TorchStructureFound found;
  if (torch_structure_find(a, found) == nullptr) return nullptr;
  torch_structure_witness(*found.entry, a, *found.positions, *found.coordinates);
  return found.entry->state.ptr();
}

// Clears the entry `found` names if the arrays have been written since it was given its record,
// and records them as they are now (see the top). Nothing for an entry without a record.
inline void torch_structure_revalidate(const TorchStructureFound& found) {
  TorchStructureEntry& entry = *found.entry;
  if (!entry.record || entry.record->current(*found.positions, *found.coordinates)) return;
  // A new record before the dict is cleared: clearing frees plans and records, whose destructors
  // must find the entry consistent.
  entry.record =
      std::make_unique<TorchStructureRecord>(*found.positions, *found.coordinates);
  entry.sighted.clear();
  if (entry.state) PyDict_Clear(entry.state.ptr());
}

// Drops the entries whose arrays are gone -- those that never got weak references -- and returns
// how many remain. Run when the registry has doubled since the last sweep, so its cost is constant
// per entry added.
inline size_t torch_structure_sweep() {
  TorchStructures& r = torch_structures();
  for (auto it = r.entries.begin(); it != r.entries.end();) {
    auto next = std::next(it);
    if (!it->second.live()) torch_structure_erase(it);
    it = next;
  }
  r.graveyard.clear();
  r.swept_at = r.entries.size();
  return r.entries.size();
}

// `a`'s structure's entry, added when there is none (revalidating one there is); nullptr with no
// error set if `a` has no storage to key on. An entry added here holds no Python object and nothing
// that records `a`; torch_structure_create gives it what Python needs.
inline TorchStructureEntry* torch_structure_entry(const at::Tensor& a, TorchStructureFound& found) {
  TorchStructureKey key;
  torch_structure_find(a, found, key);
  if (found.positions == nullptr) return nullptr;  // no storage to key on
  if (found.entry != nullptr) {
    torch_structure_revalidate(found);
    return found.entry;
  }
  TorchStructures& r = torch_structures();
  r.graveyard.clear();
  if (r.entries.size() >= 2 * std::max<size_t>(r.swept_at, 64)) torch_structure_sweep();
  TorchStructureEntry entry;
  entry.positions_alive = c10::weak_intrusive_ptr<c10::StorageImpl>(
      c10::intrusive_ptr<c10::StorageImpl>::reclaim_copy(
          found.positions->storage().unsafeGetStorageImpl()));
  entry.coordinates_alive = c10::weak_intrusive_ptr<c10::StorageImpl>(
      c10::intrusive_ptr<c10::StorageImpl>::reclaim_copy(
          found.coordinates->storage().unsafeGetStorageImpl()));
  entry.id = r.next_id++;
  r.by_id.emplace(entry.id, key);
  found.entry = &r.entries.emplace(key, std::move(entry)).first->second;
  return found.entry;
}

// plan.py's key for a sighting, ((b rows, b cols), b dtype, generation), as planned_matmul.h
// builds it; empty with an error set if it cannot be built.
inline py::object torch_sighting_key(const TorchSighting& x) {
  py::object shape = py::reinterpret_steal<py::object>(PyTuple_New(x.dim));
  if (!shape) return py::object();
  for (int64_t d = 0; d < x.dim; ++d) {
    PyObject* extent = PyLong_FromLongLong(x.extent[d]);
    if (extent == nullptr) return py::object();
    PyTuple_SET_ITEM(shape.ptr(), d, extent);
  }
  py::object generation = py::reinterpret_steal<py::object>(PyLong_FromLongLong(x.generation));
  if (!generation) return py::object();
  PyObject* dtype =
      reinterpret_cast<PyObject*>(torch::getTHPDtype(static_cast<at::ScalarType>(x.dtype)));
  return py::reinterpret_steal<py::object>(
      PyTuple_Pack(3, shape.ptr(), dtype, generation.ptr()));
}

// `entry`'s state dict (borrowed), made if it has none, with the products it has sighted moved into
// plan.py's SEEN set. nullptr with no error set if it cannot be made. A sighting that cannot be
// moved is dropped: it only means a plan waits one more call.
inline PyObject* torch_structure_state_of(TorchStructureEntry& entry) {
  if (entry.state) return entry.state.ptr();
  py::object state = py::reinterpret_steal<py::object>(PyDict_New());
  if (!state) {
    PyErr_Clear();
    return nullptr;
  }
  const py::object& seen_attr = torch_structures().seen_attr;
  if (!entry.sighted.empty() && seen_attr) {
    py::object seen = py::reinterpret_steal<py::object>(PySet_New(nullptr));
    bool ok = static_cast<bool>(seen);
    for (const TorchSighting& sighting : entry.sighted) {
      if (!ok) break;
      py::object key = torch_sighting_key(sighting);
      ok = key && PySet_Add(seen.ptr(), key.ptr()) == 0;
    }
    if (!ok || PyDict_SetItem(state.ptr(), seen_attr.ptr(), seen.ptr()) < 0) PyErr_Clear();
  }
  entry.sighted.clear();
  entry.state = std::move(state);
  return entry.state.ptr();
}

// The state dict of `a`'s structure (borrowed) as Python reads it, recording `a` as a witness;
// nullptr with no error set when it has none. Python sees what it would have, had the C entry kept
// its sightings in the dict itself: an entry with sightings and no dict gets one, holding them.
inline PyObject* torch_structure_python_state(const at::Tensor& a) {
  TorchStructureFound found;
  if (torch_structure_find(a, found) == nullptr) return nullptr;
  torch_structure_witness(*found.entry, a, *found.positions, *found.coordinates);
  if (!found.entry->state && found.entry->sighted.empty()) return nullptr;
  return torch_structure_state_of(*found.entry);
}

// The state dict of `a`'s structure (borrowed) for Python to keep something in, adding the entry
// and making the dict as needed; nullptr with no error set if one cannot be made. Records `a` as a
// witness, gives the entry the weak references that drop it with its arrays, and its record.
inline PyObject* torch_structure_create(const at::Tensor& a) {
  TorchStructureFound found;
  TorchStructureEntry* entry = torch_structure_entry(a, found);
  if (entry == nullptr) return nullptr;
  PyObject* state = torch_structure_state_of(*entry);
  if (state == nullptr) return nullptr;
  torch_structure_witness(*entry, a, *found.positions, *found.coordinates);
  torch_structure_attach(*entry, *found.positions, *found.coordinates);
  if (!entry->record)
    entry->record = std::make_unique<TorchStructureRecord>(*found.positions, *found.coordinates);
  return state;
}

// The C entry's first sighting of the product `sighting` on `a`'s structure. An entry with a state
// dict is returned (borrowed) for the caller to record the sighting in its SEEN set, as plan.py
// does; otherwise the sighting is kept as plain data, at most `bound` of them, and nullptr is
// returned. Nothing else is made: an operand built over new arrays for every call only ever makes
// first sightings, so they are most of what such a call costs. The entry holds no Python object,
// records no witness and has no weak references; a sweep drops it once its arrays are gone.
inline PyObject* torch_structure_sight(const at::Tensor& a, const TorchSighting& sighting,
                                       size_t bound) {
  TorchStructureFound found;
  TorchStructureEntry* entry = torch_structure_entry(a, found);
  if (entry == nullptr) return nullptr;
  if (entry->state) return entry->state.ptr();
  auto& sighted = entry->sighted;
  if (sighted.size() < bound && std::find(sighted.begin(), sighted.end(), sighting) == sighted.end())
    sighted.push_back(sighting);
  return nullptr;
}

// Whether the C entry has sighted `sighting` on this entry (and not yet moved it into a state dict).
inline bool torch_structure_sighted(const TorchStructureEntry& entry,
                                    const TorchSighting& sighting) {
  return std::find(entry.sighted.begin(), entry.sighted.end(), sighting) != entry.sighted.end();
}

// For tests: how many structures have an entry.
inline size_t torch_structure_count() { return torch_structures().entries.size(); }

}  // namespace scorch_native
