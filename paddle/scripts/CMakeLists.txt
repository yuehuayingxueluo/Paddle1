configure_file(submit_local.sh.in paddle @ONLY)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/paddle
  DESTINATION bin
  PERMISSIONS
    OWNER_EXECUTE
    OWNER_WRITE
    OWNER_READ
    GROUP_EXECUTE
    GROUP_READ
    WORLD_EXECUTE
    WORLD_READ)
