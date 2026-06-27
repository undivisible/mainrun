cmake_minimum_required(VERSION 3.20)
project(mlx_gpt CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MLX REQUIRED HINTS /opt/homebrew/share/cmake/MLX)

add_executable(train
  main.cpp
  model.cpp
  data.cpp
  inference/engine.cpp
  training/trainer.cpp
  training/optimizer.cpp
)

add_executable(bench_forward
  benchmark_forward.cpp
  model.cpp
)

target_include_directories(train PRIVATE /opt/homebrew/include ${CMAKE_SOURCE_DIR})
target_include_directories(bench_forward PRIVATE /opt/homebrew/include ${CMAKE_SOURCE_DIR})
target_link_directories(train PRIVATE /opt/homebrew/lib)
target_link_directories(bench_forward PRIVATE /opt/homebrew/lib)
target_link_libraries(train PRIVATE mlx)
target_link_libraries(bench_forward PRIVATE mlx)
target_compile_options(train PRIVATE -O2 -Wall -Wextra)
target_compile_options(bench_forward PRIVATE -O2 -Wall -Wextra)
