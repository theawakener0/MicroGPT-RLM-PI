#pragma once
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>

namespace microgpt {

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

class Logger {
public:
    static void set_level(LogLevel level) {
        current_level = level;
    }
    
    static void debug(const std::string& msg) {
        log(LogLevel::DEBUG, "[DEBUG]", msg);
    }
    
    static void info(const std::string& msg) {
        log(LogLevel::INFO, "[INFO]", msg);
    }
    
    static void warning(const std::string& msg) {
        log(LogLevel::WARNING, "[WARN]", msg);
    }
    
    static void error(const std::string& msg) {
        log(LogLevel::ERROR, "[ERROR]", msg);
    }
    
private:
    static LogLevel current_level;
    
    static void log(LogLevel level, const std::string& prefix, const std::string& msg) {
        if (level < current_level) return;
        
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        
        std::cout << prefix << " " 
                  << std::put_time(std::localtime(&time), "%H:%M:%S") 
                  << " | " << msg << std::endl;
    }
};

}
