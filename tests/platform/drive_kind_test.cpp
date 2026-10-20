// drive_kind_test: which kind of drive a path is on (Linux: sysfs, with partitions and LVM volumes followed down to
// their disk). The two paths are this PC's: the NVMe that holds the model packs, and /home (LVM on a SATA SSD). A path
// that does not exist is skipped; a path that does and gets the wrong kind fails.
#include "strata/platform/direct_file.hpp"

#include <cstdio>
#include <filesystem>

using namespace strata::platform;

namespace {
int failures = 0;

const char* name(DriveKind k) {
    switch (k) {
        case DriveKind::Nvme: return "Nvme";
        case DriveKind::Ssd: return "Ssd";
        case DriveKind::Hdd: return "Hdd";
        default: return "Unknown";
    }
}

void expect(const char* path, DriveKind want) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        std::printf("skip %s (missing)\n", path);
        return;
    }
    const DriveKind got = drive_kind(path);
    std::printf("%s: %s (want %s)\n", path, name(got), name(want));
    if (got != want) {
        std::fprintf(stderr, "FAILED %s:%d: drive_kind(%s) is %s, want %s\n", __FILE__, __LINE__, path, name(got),
                     name(want));
        ++failures;
    }
}
}  // namespace

int main() {
    expect("/mnt/nvme/strata-data/packs/iq3_s/experts.bin", DriveKind::Nvme);
    expect("/home/models", DriveKind::Ssd);
    if (drive_kind("/no/such/path/for/drive_kind") != DriveKind::Unknown) {
        std::fprintf(stderr, "FAILED %s:%d: a missing path is not Unknown\n", __FILE__, __LINE__);
        ++failures;
    }
    std::printf("%s\n", failures == 0 ? "drive_kind_test: ok" : "drive_kind_test: FAILED");
    return failures == 0 ? 0 : 1;
}
