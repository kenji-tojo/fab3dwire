#pragma once

#include <chrono>


namespace wiregrad {

class Timer {
private:
    using Clock_ = std::chrono::high_resolution_clock;
    std::chrono::time_point<Clock_> start = Clock_::now();

public:
    void reset()
    {
        start = Clock_::now();
    }

    [[nodiscard]] double elapsed_microseconds() const
    {
        const auto end = Clock_::now();
        const auto ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return static_cast<double>(ms.count());
    }

};

} // namespace wiregrad



