# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "cm: 1 messages, 0 services")

set(MSG_I_FLAGS "-Icm:Z:/matlab_msg_gen_ros1/win64/src/cm/msg;-Istd_msgs:C:/Program Files/MATLAB/R2023b/sys/ros1/win64/ros1/share/std_msgs/cmake/../msg;-Istd_msgs:C:/Program Files/MATLAB/R2023b/sys/ros1/win64/ros1/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(cm_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "Z:/matlab_msg_gen_ros1/win64/src/cm/msg/msg_cm.msg" NAME_WE)
add_custom_target(_cm_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "cm" "Z:/matlab_msg_gen_ros1/win64/src/cm/msg/msg_cm.msg" "std_msgs/Header"
)

#
#  langs = gencpp;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(cm
  "Z:/matlab_msg_gen_ros1/win64/src/cm/msg/msg_cm.msg"
  "${MSG_I_FLAGS}"
  "C:/Program Files/MATLAB/R2023b/sys/ros1/win64/ros1/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/cm
)

### Generating Services

### Generating Module File
_generate_module_cpp(cm
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/cm
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(cm_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(cm_generate_messages cm_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "Z:/matlab_msg_gen_ros1/win64/src/cm/msg/msg_cm.msg" NAME_WE)
add_dependencies(cm_generate_messages_cpp _cm_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(cm_gencpp)
add_dependencies(cm_gencpp cm_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS cm_generate_messages_cpp)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(cm
  "Z:/matlab_msg_gen_ros1/win64/src/cm/msg/msg_cm.msg"
  "${MSG_I_FLAGS}"
  "C:/Program Files/MATLAB/R2023b/sys/ros1/win64/ros1/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/cm
)

### Generating Services

### Generating Module File
_generate_module_py(cm
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/cm
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(cm_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(cm_generate_messages cm_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "Z:/matlab_msg_gen_ros1/win64/src/cm/msg/msg_cm.msg" NAME_WE)
add_dependencies(cm_generate_messages_py _cm_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(cm_genpy)
add_dependencies(cm_genpy cm_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS cm_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/cm)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/cm
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(cm_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(cm_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/cm)
  install(CODE "execute_process(COMMAND \"C:/Users/Zhiguo/AppData/Roaming/MathWorks/MATLAB/R2023b/ros1/win64/venv/Scripts/python.exe\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/cm\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/cm
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(cm_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(cm_generate_messages_py std_msgs_generate_messages_py)
endif()
