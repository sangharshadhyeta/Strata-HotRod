// Strata. The DeepSeek engine's decode timeline (src/dsv41/ds4_timeline.hpp): off until the CTL file says "1"; whole windows
// only; each chunk repeats the window of the write before it; reads land with their layer and kind.  No GPU, no model.
#include "ds4_timeline.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

namespace tl = strata::ds4::tl;

static int fails = 0;
#define CHECK(c)                                                                     \
    do {                                                                             \
        if (!(c)) {                                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                 \
            ++fails;                                                                 \
        }                                                                            \
    } while (0)

static std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
static void put(const std::string& p, const char* s) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f << s;
}
static int count(const std::string& s, const std::string& what) {
    int n = 0;
    for (size_t at = s.find(what); at != std::string::npos; at = s.find(what, at + 1)) ++n;
    return n;
}
// one decode window: layers 0..n-1, each stamped (a window starts when the layer number falls)
static void window(int n) {
    for (int li = 0; li < n; ++li) {
        int64_t t[7];
        for (int j = 0; j < 7; ++j) t[j] = tl::ns();
        tl::layer(li, 1, li % 2, 3, t);
    }
}

int main() {
    const std::string dir = std::filesystem::temp_directory_path().string();
    const std::string ctl = dir + "/ds4-timeline-test.ctl", out = dir + "/ds4-timeline-test.json";
    std::filesystem::remove(out);

    put(ctl, "0");
    tl::tick(ctl.c_str(), out.c_str());
    CHECK(!tl::on());
    CHECK(!std::filesystem::exists(out));

    put(ctl, "1");
    tl::tick(ctl.c_str(), out.c_str());          // switches on, writes nothing yet
    CHECK(tl::on());
    CHECK(!std::filesystem::exists(out));

    window(40);                                  // window 0 (still filling: no later window has started)
    tl::tick(ctl.c_str(), out.c_str());
    CHECK(!std::filesystem::exists(out));

    window(40);                                  // window 1 starts: window 0 is whole
    tl::read(5, true, tl::ns());
    tl::read(6, false, tl::ns());
    tl::tick(ctl.c_str(), out.c_str());
    std::string js = slurp(out);
    CHECK(js.find("\"v\":2") != std::string::npos);
    CHECK(js.find("\"first\":0,\"last\":0") != std::string::npos);
    CHECK(count(js, "],[") + 1 >= 40);           // the 40 layer rows of window 0 (and no row of window 1)
    CHECK(js.find("[0,39,1,") != std::string::npos);
    CHECK(js.find("[1,0,1,") == std::string::npos);

    window(40);                                  // window 2 starts: window 1 is whole
    tl::tick(ctl.c_str(), out.c_str());
    js = slurp(out);
    CHECK(js.find("\"first\":0,\"last\":1") != std::string::npos);   // the write before's window again, then the new one
    CHECK(js.find("[1,0,1,") != std::string::npos);
    CHECK(js.find(",5,1,") != std::string::npos);   // the demand read of layer 5 (worker, layer, demand, t0, t1)
    CHECK(js.find(",6,0,") != std::string::npos);   // the guess of layer 6

    tl::tick(ctl.c_str(), out.c_str());          // nothing new: the file stays as it is
    CHECK(slurp(out) == js);

    put(ctl, "0");
    tl::tick(ctl.c_str(), out.c_str());          // switched off: the file goes
    CHECK(!tl::on());
    CHECK(!std::filesystem::exists(out));

    std::filesystem::remove(ctl);
    if (fails == 0) std::printf("timeline_test: all passed\n");
    return fails == 0 ? 0 : 1;
}
