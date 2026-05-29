// Log.h. Central render-logging sink with two independent, composable
// outputs that default to off (quiet):
//   verbose  (--verbose)            echo chatter inline to stdout
//   file     (--debug / --log-path) capture everything to a sidecar log
// Quiet is the default (neither on). Result lines (Wrote / render took)
// always reach stdout regardless of mode, and are additionally captured
// to the file whenever the file sink is on.
//
// Call sites stream exactly like std::cout, so the change is minimal:
//   PCR_LOG    << "per-pass chatter ...";   // stdout iff verbose, file iff fileSink
//   PCR_RESULT << "Render took " << ms;      // always stdout, plus file iff fileSink
// At the end of a render, once the final image path is known:
//   pcr::logging::flush(outputPath);         // writes the buffer beside the image
//
// All four render backends (CPU, Metal, OpenGL) and both front-ends
// (CLI, GUI) include this header and share the one global state, so the
// behavior is uniform across every implementation.
#pragma once

#include <fstream>
#include <ios>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace pcr {
namespace logging {

// Runtime state. Function-local statics keep this header-only and free
// of static-initialization-order issues across translation units.
inline bool &verboseInline() { static bool v = false; return v; }   // --verbose
inline bool &fileSink()      { static bool f = false; return f; }   // --debug / --log-path
inline std::string &logPathOverride() { static std::string p; return p; }
inline std::ostringstream &buffer()   { static std::ostringstream b; return b; }
inline std::mutex &mutex_()           { static std::mutex m; return m; }

// One streamed statement. Accumulates into a local buffer and routes the
// finished text to the selected sinks on destruction (end of the full
// expression), so a whole `<<` chain is emitted atomically under the lock.
class Line {
public:
    explicit Line(bool alwaysToStdout)
        : toStdout_(alwaysToStdout || verboseInline()), toFile_(fileSink()) {}
    ~Line() {
        const std::string s = ss_.str();
        if (s.empty() || (!toStdout_ && !toFile_)) return;
        std::lock_guard<std::mutex> g(mutex_());
        if (toStdout_) { std::cout << s; std::cout.flush(); }
        if (toFile_)   { buffer() << s; }
    }
    template <class T> Line &operator<<(const T &v) { ss_ << v; return *this; }
    // Stream manipulators such as std::endl and std::flush.
    Line &operator<<(std::ostream &(*m)(std::ostream &)) { ss_ << m; return *this; }
private:
    std::ostringstream ss_;
    bool toStdout_;
    bool toFile_;
};

inline Line info()   { return Line(false); }   // chatter, gated by verbose
inline Line result() { return Line(true);  }   // Wrote / render took, always stdout

// Write the captured buffer to the sidecar log and clear it. No-op when
// the file sink is off. imagePath is the just-written image; the log
// lands beside it (".png" becomes ".log") unless --log-path set a path.
inline void flush(const std::string &imagePath) {
    if (!fileSink()) return;
    std::string path = logPathOverride();
    if (path.empty()) {
        path = imagePath;
        const std::string::size_type pos = path.rfind(".png");
        if (pos != std::string::npos) path.replace(pos, 4, ".log");
        else path += ".log";
    }
    std::lock_guard<std::mutex> g(mutex_());
    std::ofstream f(path, std::ios::trunc);
    if (f) f << buffer().str();
    buffer().str(std::string());
    buffer().clear();
    // Tell the user where the log went (stdout only, the buffer is spent).
    std::cout << "Wrote log " << path << std::endl;
}

} // namespace logging
} // namespace pcr

#define PCR_LOG    ::pcr::logging::info()
#define PCR_RESULT ::pcr::logging::result()
