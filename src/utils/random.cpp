#include "random.hpp"

namespace microgpt {
std::mt19937_64 Random::gen{std::random_device{}()};
}
