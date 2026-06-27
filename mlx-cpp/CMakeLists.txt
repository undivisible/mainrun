cmake_minimum_required(VERSION 3.20)
project(mlx_gpt CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MLX REQUIRED HINTS /opt/homebrew/share/cmake/MLX)

add_executable(train
  main.cpp
  model.cpp
  optimizer.cpp
  data.cpp
  training.cpp
  inference.cpp
)

target_include_directories(train PRIVATE /opt/homebrew/include)
target_link_directories(train PRIVATE /opt/homebrew/lib)
target_link_libraries(train PRIVATE mlx)
target_compile_options(train PRIVATE -O2 -Wall -Wextra)
