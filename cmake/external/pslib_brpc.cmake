# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(ExternalProject)

set(PSLIB_BRPC_PROJECT "extern_pslib_brpc")
if((NOT DEFINED PSLIB_BRPC_NAME) OR (NOT DEFINED PSLIB_BRPC_URL))
  message(STATUS "use pre defined download url")
  set(PSLIB_BRPC_VER
      "0.1.0"
      CACHE STRING "" FORCE)
  set(PSLIB_BRPC_NAME
      "pslib_brpc"
      CACHE STRING "" FORCE)
  set(PSLIB_BRPC_URL
      "https://pslib.bj.bcebos.com/pslib_brpc.tar.gz"
      CACHE STRING "" FORCE)
endif()
message(
  STATUS
    "PSLIB_BRPC_NAME: ${PSLIB_BRPC_NAME}, PSLIB_BRPC_URL: ${PSLIB_BRPC_URL}")
set(PSLIB_BRPC_PREFIX_DIR "${THIRD_PARTY_PATH}/pslib_brpc")
set(PSLIB_BRPC_DOWNLOAD_DIR
    "${PSLIB_BRPC_PREFIX_DIR}/src/${PSLIB_BRPC_PROJECT}")
set(PSLIB_BRPC_DST_DIR "pslib_brpc")
set(PSLIB_BRPC_INSTALL_ROOT "${THIRD_PARTY_PATH}/install")
set(PSLIB_BRPC_INSTALL_DIR ${PSLIB_BRPC_INSTALL_ROOT}/${PSLIB_BRPC_DST_DIR})
set(PSLIB_BRPC_ROOT ${PSLIB_BRPC_INSTALL_DIR})
set(PSLIB_BRPC_INC_DIR ${PSLIB_BRPC_ROOT}/include)
set(PSLIB_BRPC_LIB_DIR ${PSLIB_BRPC_ROOT}/lib)
set(PSLIB_BRPC_LIB ${PSLIB_BRPC_LIB_DIR}/libbrpc.a)
set(PSLIB_BRPC_IOMP_LIB ${PSLIB_BRPC_LIB_DIR}/libiomp5.so) #todo what is this
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${PSLIB_BRPC_ROOT}/lib")

include_directories(${PSLIB_BRPC_INC_DIR})

file(
  WRITE ${PSLIB_BRPC_DOWNLOAD_DIR}/CMakeLists.txt
  "PROJECT(PSLIB_BRPC)\n" "cmake_minimum_required(VERSION 3.0)\n"
  "install(DIRECTORY ${PSLIB_BRPC_NAME}/include ${PSLIB_BRPC_NAME}/lib \n"
  "        DESTINATION ${PSLIB_BRPC_DST_DIR})\n")

ExternalProject_Add(
  ${PSLIB_BRPC_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  PREFIX ${PSLIB_BRPC_PREFIX_DIR}
  DOWNLOAD_DIR ${PSLIB_BRPC_DOWNLOAD_DIR}
  DOWNLOAD_COMMAND
    wget --no-check-certificate ${PSLIB_BRPC_URL} -c -q -O
    ${PSLIB_BRPC_NAME}.tar.gz && tar zxvf ${PSLIB_BRPC_NAME}.tar.gz
  DOWNLOAD_NO_PROGRESS 1
  UPDATE_COMMAND ""
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${PSLIB_BRPC_INSTALL_ROOT}
             -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
  CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${PSLIB_BRPC_INSTALL_ROOT}
                   -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
  BUILD_BYPRODUCTS ${PSLIB_BRPC_LIB})

add_library(pslib_brpc SHARED IMPORTED GLOBAL)
set_property(TARGET pslib_brpc PROPERTY IMPORTED_LOCATION ${PSLIB_BRPC_LIB})
add_dependencies(pslib_brpc ${PSLIB_BRPC_PROJECT})
