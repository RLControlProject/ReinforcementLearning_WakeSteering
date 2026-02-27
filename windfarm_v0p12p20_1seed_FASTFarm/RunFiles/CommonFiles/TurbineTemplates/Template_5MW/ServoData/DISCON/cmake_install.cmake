# Install script for directory: /projects/rlfarmcontr/RLControl_Parallel/RLControl/FFSetup/Template_5MW/5MW_Baseline/ServoData/DISCON

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/opt/cray/pe/cce/17.0.0/binutils/x86_64/x86_64-pc-linux-gnu/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/projects/rlfarmcontr/RLControl_Parallel/RLControl/FFSetup/Template_5MW/5MW_Baseline/ServoData/DISCON/../DISCON.dll" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/projects/rlfarmcontr/RLControl_Parallel/RLControl/FFSetup/Template_5MW/5MW_Baseline/ServoData/DISCON/../DISCON.dll")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/projects/rlfarmcontr/RLControl_Parallel/RLControl/FFSetup/Template_5MW/5MW_Baseline/ServoData/DISCON/../DISCON.dll"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/projects/rlfarmcontr/RLControl_Parallel/RLControl/FFSetup/Template_5MW/5MW_Baseline/ServoData/DISCON/../DISCON.dll")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/projects/rlfarmcontr/RLControl_Parallel/RLControl/FFSetup/Template_5MW/5MW_Baseline/ServoData/DISCON/.." TYPE SHARED_LIBRARY FILES "/projects/rlfarmcontr/RLControl_Parallel/RLControl/FFSetup/Template_5MW/5MW_Baseline/ServoData/DISCON/DISCON.dll")
  if(EXISTS "$ENV{DESTDIR}/projects/rlfarmcontr/RLControl_Parallel/RLControl/FFSetup/Template_5MW/5MW_Baseline/ServoData/DISCON/../DISCON.dll" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/projects/rlfarmcontr/RLControl_Parallel/RLControl/FFSetup/Template_5MW/5MW_Baseline/ServoData/DISCON/../DISCON.dll")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/cray/pe/cce/17.0.0/binutils/x86_64/x86_64-pc-linux-gnu/bin/strip" "$ENV{DESTDIR}/projects/rlfarmcontr/RLControl_Parallel/RLControl/FFSetup/Template_5MW/5MW_Baseline/ServoData/DISCON/../DISCON.dll")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/projects/rlfarmcontr/RLControl_Parallel/RLControl/FFSetup/Template_5MW/5MW_Baseline/ServoData/DISCON/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
