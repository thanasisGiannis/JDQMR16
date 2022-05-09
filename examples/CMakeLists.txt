cmake_minimum_required(VERSION 3.16)

project(ex_jdqmr16)

set(SRC ex_eigs.cu
        include/mmio.cu 
)



add_executable(ex_jdqmr16 ${SRC})
set_target_properties(ex_jdqmr16 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

target_include_directories(ex_jdqmr16 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_link_directories(ex_jdqmr16 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../lib")
#target_link_directories(ex_jdqmr16 PUBLIC "${MKLROOT}/lib/intel64")



target_link_libraries(ex_jdqmr16 PUBLIC
											#-lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core 
											#-lmkl_blacs_intelmpi_ilp64 
                                 -lm -lcusolver -lcusparse -lcurand
                                 -lcublas -lculibos -lcudart -lpthread -ldl
                                 -lgomp -lpthread -lm -ldl
											libjdqmr16.so
											)

