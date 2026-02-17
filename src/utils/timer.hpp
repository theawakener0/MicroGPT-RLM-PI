#pragma once
#include <chrono>
#include <string>
#include <map>
#include <iostream>

namespace microgpt {

class Timer {
public:
    static void start(const std::string& name) {
        timings[name] = std::chrono::high_resolution_clock::now();
    }
    
    static float stop(const std::string& name) {
        auto now = std::chrono::high_resolution_clock::now();
        auto it = timings.find(name);
        if (it == timings.end()) return 0.0f;
        
        float ms = std::chrono::duration<float, std::milli>(now - it->second).count();
        timings.erase(it);
        return ms;
    }
    
    static void print_elapsed(const std::string& name) {
        auto now = std::chrono::high_resolution_clock::now();
        auto it = timings.find(name);
        if (it == timings.end()) {
            std::cout << "Timer '" << name << "' not found" << std::endl;
            return;
        }
        
        float ms = std::chrono::duration<float, std::milli>(now - it->second).count();
        std::cout << name << ": " << ms << " ms" << std::endl;
    }
    
private:
    static std::map<std::string, std::chrono::high_resolution_clock::time_point> timings;
};

}
