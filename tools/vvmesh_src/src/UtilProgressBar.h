/* (The MIT License)

Copyright (c) 2016 Prakhar Srivastav <prakhar@prakhar.me>

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
'Software'), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 MODIFIED BY SXSONG(@qq.com)
 */

#ifndef PROGRESSBAR_PROGRESSBAR_HPP
#define PROGRESSBAR_PROGRESSBAR_HPP

#include <chrono>
#include <iostream>
#include <thread>
#include <mutex>

// https://stackoverflow.com/a/31392259
class timing_context {
 public:
  typedef std::chrono::duration<float> sec;
  typedef std::chrono::duration<float, std::milli> msec;
  typedef std::chrono::duration<float, std::micro> usec;
  typedef usec duration;
  std::map<std::string, duration> timings;
  static std::string format_duration(
      const duration& d) {
    float v;
    v = std::chrono::duration_cast<sec>(d).count();
    if (v >= 1) return std::to_string(v) + " s";
    v = std::chrono::duration_cast<msec>(d).count();
    if (v >= 1) return std::to_string(v) + " ms";
    v = std::chrono::duration_cast<usec>(d).count();
    return std::to_string(v) + " us";
  }
};

class timer {
 public:
  timer(timing_context& ctx, std::string name)
      : ctx(ctx), name(name), start(std::chrono::steady_clock::now()) {}

  ~timer() { ctx.timings[name] = std::chrono::steady_clock::now() - start; }

  timing_context& ctx;
  std::string name;
  std::chrono::steady_clock::time_point start;
};

class ProgressBar {
 private:
  std::string header;
  unsigned int ticks = 0;

  const unsigned int total_ticks;
  const unsigned int bar_width;
  const char complete_char = '=';
  const char incomplete_char = ' ';
  const std::chrono::steady_clock::time_point start_time =
      std::chrono::steady_clock::now();

  std::mutex _mutex;
  std::chrono::steady_clock::time_point _last_refresh_t;
  float _slient_duration;

 public:
  ProgressBar(std::string header, unsigned int total, unsigned int width,
              char complete, char incomplete, float fps = 1)
      : header{header},
        total_ticks{total},
        bar_width{width},
        complete_char{complete},
        incomplete_char{incomplete} {
    _last_refresh_t = std::chrono::steady_clock::now();
    _slient_duration = 1.f / fps;  // milisecond duration
  }

  ProgressBar(std::string header, unsigned int total, unsigned int width)
      : header{header}, total_ticks{total}, bar_width{width} {
    _last_refresh_t = std::chrono::steady_clock::now();
    _slient_duration = 1.f;  // milisecond duration
  }

  unsigned int operator++() {
    std::lock_guard<std::mutex> guard(_mutex);
    return ++ticks;
  }

  void display() {
    if (_mutex.try_lock()) {
      auto now = std::chrono::steady_clock::now();
      std::chrono::seconds time_span =
          std::chrono::duration_cast<std::chrono::seconds>(now -
                                                           _last_refresh_t);
      if (time_span.count() > _slient_duration) {
        _last_refresh_t = now;
        float progress = (float)ticks / total_ticks;
        int pos = (int)(bar_width * progress);

        std::chrono::steady_clock::time_point now =
            std::chrono::steady_clock::now();
        auto time_elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(now -
                                                                  start_time)
                .count();
        auto estimate_eta = (1.0f - progress) / progress * time_elapsed;
        std::cout << header << " [";

        for (unsigned i = 0; i < bar_width; ++i) {
          if (i < pos)
            std::cout << complete_char;
          else if (i == pos)
            std::cout << ">";
          else
            std::cout << incomplete_char;
        }

        std::cout << "] " << int(progress * 100.0) << "% "
                  << float(time_elapsed) / 1000.0 << "s( ETA "
                  << estimate_eta / 1000.0 << "s)...\r";
        std::cout.flush();
      }
      _mutex.unlock();
    }
  }

  void done() {
    display();
    std::cout << std::endl;
  }
};

#endif  // PROGRESSBAR_PROGRESSBAR_HPP
