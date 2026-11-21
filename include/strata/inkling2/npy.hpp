// include/strata/inkling2/npy.hpp - a reader for the reference's seam dumps: float32, C-order, little-endian .npy (version 1 or 2).
// Test support for docs/INKLING-PORT.md (the dumps come from Maya's ref/inkling2.py); header-only, no dependencies.
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata {
namespace inkling2 {

/// The file's values; `shape` receives its dimensions.  Throws when the file is missing or not little-endian float32 C-order.
inline std::vector<float> load_npy(const std::string& path, std::vector<int64_t>& shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[10];
    f.read(magic, 10);
    if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error(path + ": not an npy file");
    uint32_t hl = (unsigned char) magic[8] | ((unsigned char) magic[9] << 8);
    if ((unsigned char) magic[6] >= 2) {   // version 2: a 4-byte header length
        char more[2];
        f.read(more, 2);
        hl |= ((uint32_t) (unsigned char) more[0] << 16) | ((uint32_t) (unsigned char) more[1] << 24);
    }
    std::string hdr(hl, ' ');
    f.read(&hdr[0], hl);
    if (hdr.find("'<f4'") == std::string::npos || hdr.find("False") == std::string::npos)
        throw std::runtime_error(path + ": not little-endian float32 C-order: " + hdr);
    shape.clear();
    const size_t a = hdr.find('('), b = hdr.find(')');
    std::stringstream ss(hdr.substr(a + 1, b - a - 1));
    std::string tok;
    int64_t n = 1;
    while (std::getline(ss, tok, ','))
        if (tok.find_first_of("0123456789") != std::string::npos) { shape.push_back(std::stoll(tok)); n *= shape.back(); }
    std::vector<float> m((size_t) n);
    f.read(reinterpret_cast<char*>(m.data()), (std::streamsize) (n * 4));
    if (!f) throw std::runtime_error(path + ": truncated");
    return m;
}

inline std::vector<float> load_npy(const std::string& path) {
    std::vector<int64_t> shape;
    return load_npy(path, shape);
}

}  // namespace inkling2
}  // namespace strata
