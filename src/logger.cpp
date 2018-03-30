#include "logger.hpp"

void initialize_logging() {
    auto console = spdlog::stdout_color_mt("LOG");
}
