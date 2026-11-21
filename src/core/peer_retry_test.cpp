// peer_retry_test.cpp - the peer reserve retry's sizing rule (include/strata/core/peer_retry.hpp).  CPU only.
#include "strata/core/peer_retry.hpp"

#include <cstdio>

using strata::core::next_peer_reserve_mib;

static int g_fail = 0;
#define CHECK(c, ...)                                                      \
    do {                                                                   \
        if (!(c)) {                                                        \
            ++g_fail;                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s  ", __FILE__, __LINE__, #c); \
            std::fprintf(stderr, __VA_ARGS__);                             \
            std::fprintf(stderr, "\n");                                    \
        }                                                                  \
    } while (0)

int main() {
    // the community report: 600 was not enough and 1200 was; a shortfall of a few hundred MiB reaches it in one or two attempts
    CHECK(next_peer_reserve_mib(600, 300, 1) == 1156, "got %d", next_peer_reserve_mib(600, 300, 1));
    CHECK(next_peer_reserve_mib(600, 0, 1) == 856, "no shortfall known: the minimum step of 256 MiB; got %d", next_peer_reserve_mib(600, 0, 1));
    // later attempts add more (the allocations after the failing one are not counted)
    int r = 600;
    for (int a = 1; a <= strata::core::kPeerRetryMaxAttempts; ++a) {
        const int n = next_peer_reserve_mib(r, 300, a);
        CHECK(n > r, "attempt %d did not raise the reserve: %d -> %d", a, r, n);
        r = n;
    }
    // the cap: a huge shortfall never asks for more than the cap, and a reserve already at the cap is unchanged
    CHECK(next_peer_reserve_mib(600, 100000, 1) == strata::core::kPeerRetryCapMiB, "cap not applied");
    CHECK(next_peer_reserve_mib(4096, 500, 1) == 4096, "at the cap the reserve must not grow");
    CHECK(next_peer_reserve_mib(5000, 500, 1) == 5000, "a reserve the user set above the cap is left alone");
    // nonsense input never lowers the reserve
    CHECK(next_peer_reserve_mib(600, -50, 0) >= 600, "negative shortfall / attempt 0 lowered the reserve");
    if (g_fail != 0) { std::fprintf(stderr, "peer_retry_test: %d check(s) failed\n", g_fail); return 1; }
    std::printf("peer_retry_test: all passed\n");
    return 0;
}
