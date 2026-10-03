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
 * WHY THIS IS NOT A DIFFERENT EXPERIMENT. The cache returns the SAME POINTER the real
 * getenv would, so every predicate reads the same value and the kernel's decisions are
 * unchanged -- this is not a stub returning NULL, which would silently switch every arm
 * to its compiled default and look like a clean null result. Any write to the
 * environment clears the cache, so an arm flip is seen on the next call. And a release
 * object performs NO getenv at all, so a build with this shim is CLOSER to what ships
 * than the hooks object is, not further from it.
 */
#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dlfcn.h>

#define CAP 512

struct slot { const char *key; char *val; int used; };
static struct slot tab[CAP];
static long n_real = 0, n_hit = 0, n_flush = 0;
static int enabled = 1;

static char *(*r_getenv)(const char *);
static int (*r_setenv)(const char *, const char *, int);
static int (*r_unsetenv)(const char *);
static int (*r_putenv)(char *);

static void resolve(void) {
    if (!r_getenv) {
        r_getenv = (char *(*)(const char *))dlsym(RTLD_NEXT, "getenv");
        r_setenv = (int (*)(const char *, const char *, int))dlsym(RTLD_NEXT, "setenv");
        r_unsetenv = (int (*)(const char *))dlsym(RTLD_NEXT, "unsetenv");
        r_putenv = (int (*)(char *))dlsym(RTLD_NEXT, "putenv");
        if (getenv("SCORCH_FASTENV_OFF")) enabled = 0;
    }
}

static unsigned long h(const char *s) {
    unsigned long x = 5381;
    while (*s) x = x * 33 + (unsigned char)*s++;
    return x;
}

static void flush(void) {
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
            s->used = 1;
            s->key = name;                 /* callers pass string literals */
            s->val = r_getenv(name);
            n_real++;
            return s->val;
        }
        if (s->key == name || strcmp(s->key, name) == 0) { n_hit++; return s->val; }
    }
    n_real++;
    return r_getenv(name);
}

int setenv(const char *n, const char *v, int o) { resolve(); flush(); return r_setenv(n, v, o); }
int unsetenv(const char *n) { resolve(); flush(); return r_unsetenv(n); }
int putenv(char *s) { resolve(); flush(); return r_putenv(s); }

long scorch_fastenv_real(void) { return n_real; }
long scorch_fastenv_hits(void) { return n_hit; }
long scorch_fastenv_flushes(void) { return n_flush; }
void scorch_fastenv_reset(void) { n_real = 0; n_hit = 0; n_flush = 0; }
int scorch_fastenv_enabled(void) { resolve(); return enabled; }
