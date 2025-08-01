cmake_minimum_required(VERSION 3.10)
project(supercool_fopt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost REQUIRED)


# Find OpenMP
find_package(OpenMP REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})

set(SOURCE_FILES
    potential/dark_abelian_higgs.cpp
    thermal_function/thermal_functions.cpp
    bounce_solver/bounce_solver_dim3.cpp
    bounce_solver/bounce_solver_dim4.cpp
    standard_cosmology/cosmology.cpp
    standard_cosmology/constant.hpp
    math_personal/sf_bessel_k.cpp 
    math_personal/sf_trig.cpp
    util/csv_importer.cpp
)

add_executable(main.exe main.cpp ${SOURCE_FILES})

target_link_libraries(main.exe Boost::boost m OpenMP::OpenMP_CXX)