#pragma once

#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>

namespace wiregrad::cpu {

template<typename Func> inline
void parallel_for(unsigned long num,
                  Func &&func,
                  unsigned int num_threads = std::thread::hardware_concurrency())
{
    auto futures = std::vector<std::future<void>>{};

    std::atomic<int> next_idx(0);
    std::atomic<bool> has_error(false);
    for (int thread_id = 0; thread_id < num_threads; thread_id++) {
        futures.emplace_back(
                std::async(std::launch::async, [&func, &next_idx, &has_error, num, thread_id]() {
                    try {
                        while (true) {
                            auto idx = next_idx.fetch_add(1);
                            if (idx >= num) break;
                            if (has_error) break;
                            func(idx, thread_id);
                        }
                    } catch (...) {
                        has_error = true;
                        throw;
                    }
                }));
    }
    for (auto &f: futures) f.get();
}

} // namespace wiregrad::cpu


