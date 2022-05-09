cmake_minimum_required(VERSION 3.16)

project(lib_jdqmr16)

set(SRC
${CMAKE_CURRENT_SOURCE_DIR}/eig/eigH.cu
${CMAKE_CURRENT_SOURCE_DIR}/eig/jdqmr16.cu
${CMAKE_CURRENT_SOURCE_DIR}/eig/restart.cu
${CMAKE_CURRENT_SOURCE_DIR}/eig/helper.cu
${CMAKE_CURRENT_SOURCE_DIR}/eig/innerSolver.cu
${CMAKE_CURRENT_SOURCE_DIR}/eig/residual.cu
${CMAKE_CURRENT_SOURCE_DIR}/eig/expandBasis.cu
${CMAKE_CURRENT_SOURCE_DIR}/eig/initBasis.cu
${CMAKE_CURRENT_SOURCE_DIR}/eig/sqmr.cu

${CMAKE_CURRENT_SOURCE_DIR}/matrix/double2halfMat.cu
)

add_library(lib_jdqmr16 SHARED ${SRC})


target_include_directories(lib_jdqmr16 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")

#target_link_directories(lib_jdqmr16 PUBLIC "${MKLROOT}/lib/intel64")
#target_include_directories(lib_jdqmr16 PUBLIC "-DMKL_ILP64 -m64 ${MKLROOT}/include/")

set_target_properties(lib_jdqmr16 PROPERTIES OUTPUT_NAME jdqmr16)
set_target_properties(lib_jdqmr16 PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../lib)

target_link_libraries(lib_jdqmr16 
                      #-g
							-lgomp -lpthread -lm -ldl
										)
