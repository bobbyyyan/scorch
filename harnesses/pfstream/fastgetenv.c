/* A behaviour-preserving fast getenv, so the per-call getenv tax can be measured
 * END-TO-END instead of decomposed into count x cost.
 *
 * WHY NOT count x cost. The count is exact (99 per SpMM call, interposed) but glibc's
 * getenv is a linear scan whose cost depends on the name: measured on this process,
 * 58 ns for an absent short name and 200 ns for the long SCORCH_SPMM_* ones, because
 * the environ holds many entries sharing that prefix and each costs a longer memcmp
 * before failing. The product therefore spans 5.7 to 19.8 us, which is wider than the
 * question it is meant to settle -- whether a measured cell sits above or below the
 * 30 us boundary in the comment at scorch_policy.h:1577.
 *
 * THE CACHE OWNS ITS KEYS. An earlier version stored the caller's `name` pointer with a
 * comment that callers pass string literals. That is a claim about EVERY caller in the
 * process -- CPython, torch, libgomp, MKL -- not about the policy header whose calls
 * prompted the shim, because LD_PRELOAD interposes process-wide. One caller passing a
 * stack or freed buffer would leave a dangling key, and the next strcmp against it could
 * match the wrong slot and return a DIFFERENT variable's value: a silently changed
 * decision, which is exactly the failure a NULL-returning stub would cause and the reason
 * one was declined. Keys are strdup'd now, which is free because insertion happens once
 * per name and the timed region is all hits.
 *
 * The borrowed pointer is still recorded, but only to be compared NUMERICALLY and never
 * dereferenced, so the run that uses the cache also reports whether the literal premise
 * actually held: if every hit is pointer-equal, every caller did pass a literal.
 *
 * WHY THIS IS NOT A DIFFERENT EXPERIMENT. The cache returns the SAME POINTER the real
 * getenv would, so every predicate reads the same value and the kernel's decisions are
 * unchanged -- this is not a stub returning NULL, which would silently switch every arm
 * to its compiled default and look like a clean null result. Any write to the
 * environment THROUGH setenv, putenv, unsetenv or clearenv clears the cache, so an arm
 * flip is seen on the next call. Direct assignment into `environ[]` is NOT intercepted and
 * cannot be; the guarantee is scoped to those four symbols rather than stated as "any
 * write", which is wider than the interception. And a release
 * object performs NO getenv at all, so a build with this shim is CLOSER to what ships
 * than the hooks object is, not further from it.
 */
#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dlfcn.h>

#define CAP 512

struct slot {
    char *key;          /* OWNED copy, the only thing ever dereferenced */
    const char *orig;   /* the caller's pointer, compared by value only, never read */
    char *val;
    int used;
};
static struct slot tab[CAP];
static long n_real = 0, n_hit_ptr = 0, n_hit_str = 0, n_flush = 0, n_evict = 0;
static int enabled = 1;

static char *(*r_getenv)(const char *);
static int (*r_setenv)(const char *, const char *, int);
static int (*r_unsetenv)(const char *);
static int (*r_putenv)(char *);
static int (*r_clearenv)(void);

static void resolve(void) {
    if (!r_getenv) {
        r_getenv = (char *(*)(const char *))dlsym(RTLD_NEXT, "getenv");
        r_setenv = (int (*)(const char *, const char *, int))dlsym(RTLD_NEXT, "setenv");
        r_unsetenv = (int (*)(const char *))dlsym(RTLD_NEXT, "unsetenv");
        r_putenv = (int (*)(char *))dlsym(RTLD_NEXT, "putenv");
        r_clearenv = (int (*)(void))dlsym(RTLD_NEXT, "clearenv");
        if (getenv("SCORCH_FASTENV_OFF")) enabled = 0;
    }
}

static unsigned long h(const char *s) {
    unsigned long x = 5381;
    while (*s) x = x * 33 + (unsigned char)*s++;
    return x;
}

static void flush(void) {
    for (int i = 0; i < CAP; i++)
        if (tab[i].used) { free(tab[i].key); tab[i].key = NULL; }
    memset(tab, 0, sizeof(tab));
    n_flush++;
}

char *getenv(const char *name) {
    resolve();
    if (!enabled) { n_real++; return r_getenv(name); }
    unsigned long i = h(name) % CAP;
    for (unsigned long p = 0; p < CAP; p++) {
        struct slot *s = &tab[(i + p) % CAP];
        if (!s->used) {
            s->key = strdup(name);
            if (!s->key) { n_real++; return r_getenv(name); }   /* fail open, not wrong */
            s->used = 1;
            s->orig = name;
            s->val = r_getenv(name);
            n_real++;
            return s->val;
        }
        /* The owned copy is what decides. The borrowed pointer only labels the hit, so a
         * caller that passed a stack buffer cannot corrupt the lookup, only the tally. */
        if (strcmp(s->key, name) == 0) {
            if (s->orig == name) n_hit_ptr++; else n_hit_str++;
            return s->val;
        }
    }
    n_evict++;                             /* table full: correct, just uncached */
    n_real++;
    return r_getenv(name);
}

int setenv(const char *n, const char *v, int o) { resolve(); flush(); return r_setenv(n, v, o); }
int unsetenv(const char *n) { resolve(); flush(); return r_unsetenv(n); }
int putenv(char *s) { resolve(); flush(); return r_putenv(s); }
int clearenv(void) { resolve(); flush(); return r_clearenv ? r_clearenv() : -1; }

long scorch_fastenv_real(void) { return n_real; }
long scorch_fastenv_hits(void) { return n_hit_ptr + n_hit_str; }
long scorch_fastenv_hits_ptr(void) { return n_hit_ptr; }
long scorch_fastenv_hits_str(void) { return n_hit_str; }
long scorch_fastenv_flushes(void) { return n_flush; }
long scorch_fastenv_evictions(void) { return n_evict; }
void scorch_fastenv_reset(void) {
    n_real = 0; n_hit_ptr = 0; n_hit_str = 0; n_flush = 0; n_evict = 0;
}
int scorch_fastenv_enabled(void) { resolve(); return enabled; }
