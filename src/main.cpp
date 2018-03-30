#include "main.hpp"

#include "cereal/archives/json.hpp"
// #include "cereal/types/eigen.hpp"
#include "logger.hpp"

#include "../cuda/structs.hpp"

#include <fstream>

int main(int argc, char* argv[]) {
    // Initialize logging
    initialize_logging();
    INFO("Initializing logging library...");

    // Test sphere structure
    scene s = {};
    
    std::ofstream os("out.json", std::ios::binary);
    cereal::JSONOutputArchive archive(os);
    archive(s);

    
    INFO("Terminating program...");
    return 0;
}
