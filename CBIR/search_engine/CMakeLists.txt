cmake_minimum_required(VERSION 3.4)
project(search_engine)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES PythonInterface.cpp Index.cpp Index.h Query.cpp Query.h)
#set(SOURCE_FILES main.cpp Index.cpp Index.h Query.cpp Query.h)
#add_executable(search_engine ${SOURCE_FILES})

#set(Boost_INCLUDE_DIR /home/fong/anaconda3/envs/python37/include)
#set(Boost_LIBRARY_DIR /home/fong/anaconda3/envs/python37/lib)

INCLUDE(FindPythonLibs)
FIND_PACKAGE(PythonInterp 3 REQUIRED)
FIND_PACKAGE(PythonLibs 3 REQUIRED)
FIND_PACKAGE(Boost COMPONENTS python3.7)

ENABLE_TESTING()
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
LINK_LIBRARIES(${Boost_LIBRARIES} ${PYTHON_LIBRARIES}) # Deprecated but so convenient!

PYTHON_ADD_MODULE(search_engine ${SOURCE_FILES})

INSTALL(TARGETS search_engine DESTINATION CBIR/search_web/)
