include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-chat.cpp
    server-chat.h
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
    server-tools.cpp
    server-tools.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ../omni)
target_include_directories(${TARGET} PRIVATE ../omni/voxcpm2)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE common omni mtmd voxcpm2_runtime ${CMAKE_THREAD_LIBS_INIT})

# llama-server-impl: server logic, reusable by app

set(TARGET llama-server-impl)

add_library(${TARGET}
    server.cpp
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ../mtmd ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC server-context llama-ui cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-server executable

set(TARGET llama-server)

add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama-server-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# llama-omni-server executable (standalone omni HTTP API)

if (LLAMA_OPENSSL)
    find_package(OpenSSL)
endif()

set(TARGET_OMNI llama-omni-server)

add_executable(${TARGET_OMNI}
    server-omni.cpp
    session.h
    session.cpp
    protocol.h
    protocol.cpp
    ws_handler.h
    ws_handler.cpp
)
install(TARGETS ${TARGET_OMNI} RUNTIME)

target_link_libraries(${TARGET_OMNI} PRIVATE omni llama-common cpp-httplib ${CMAKE_THREAD_LIBS_INIT})
if (LLAMA_OPENSSL AND OpenSSL_FOUND)
    target_link_libraries(${TARGET_OMNI} PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(${TARGET_OMNI} PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
endif()
target_compile_features(${TARGET_OMNI} PRIVATE cxx_std_17)
target_include_directories(${TARGET_OMNI} PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(${TARGET_OMNI} PRIVATE ${CMAKE_SOURCE_DIR}/vendor/cpp-httplib)
target_include_directories(${TARGET_OMNI} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)

# llama-tts-server executable (standalone VoxCPM2 TTS HTTP API)

set(TARGET_VOXCPM2 llama-tts-server)

add_executable(${TARGET_VOXCPM2} server-voxcpm2.cpp)
install(TARGETS ${TARGET_VOXCPM2} RUNTIME)

target_link_libraries(${TARGET_VOXCPM2} PRIVATE voxcpm2_runtime llama-common cpp-httplib ${CMAKE_THREAD_LIBS_INIT})
if (LLAMA_OPENSSL AND OpenSSL_FOUND)
    target_link_libraries(${TARGET_VOXCPM2} PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(${TARGET_VOXCPM2} PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
endif()
target_compile_features(${TARGET_VOXCPM2} PRIVATE cxx_std_17)
target_include_directories(${TARGET_VOXCPM2} PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(${TARGET_VOXCPM2} PRIVATE ${CMAKE_SOURCE_DIR}/vendor/cpp-httplib)
target_include_directories(${TARGET_VOXCPM2} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)
